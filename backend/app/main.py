import json
import logging
import os
import uuid
from datetime import datetime, timedelta, timezone
from pathlib import Path

from dotenv import load_dotenv

# Load the project-level .env during local development; Render uses dashboard env vars.
load_dotenv(Path(__file__).resolve().parents[1] / ".env")

from fastapi import Depends, FastAPI, File, Header, HTTPException, Request, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from . import auth
from .database import INTEGRITY_ERRORS, connect, init_db
from .extract import MAX_BYTES, MAX_FILES, ExtractError, extract_files, vision_available
from .math_agent import answer_math

logger = logging.getLogger(__name__)
app = FastAPI(title="Mathematical AI Reasoning Agent")
cors_origins = [o.strip() for o in os.getenv("CORS_ORIGINS", "http://localhost:5173").split(",") if o.strip()]
app.add_middleware(CORSMiddleware, allow_origins=cors_origins, allow_methods=["*"], allow_headers=["*"])
limiter = auth.RateLimiter()


def ts() -> str:
 # Fixed-width ISO strings so they sort and compare correctly as text.
 return datetime.now(timezone.utc).isoformat(timespec="microseconds")


@app.on_event("startup")
def startup():
 init_db()


# --------------------------------------------------------------------------- #
# Accounts
# --------------------------------------------------------------------------- #

class Register(BaseModel):
 email: str = Field(max_length=254)
 password: str = Field(max_length=auth.MAX_PASSWORD + 1)
 name: str | None = Field(default=None, max_length=60)


class Login(BaseModel):
 email: str = Field(max_length=254)
 password: str = Field(max_length=auth.MAX_PASSWORD + 1)


def client_ip(request: Request) -> str:
 forwarded = request.headers.get("x-forwarded-for", "").split(",")[0].strip()
 return forwarded or (request.client.host if request.client else "unknown")


def issue_session(c, user_id: str) -> str:
 token, token_hash = auth.new_token()
 now = datetime.now(timezone.utc)
 expires = (now + timedelta(days=auth.SESSION_DAYS)).isoformat(timespec="microseconds")
 c.execute("DELETE FROM sessions WHERE expires<?", (ts(),))  # opportunistic cleanup
 c.execute("INSERT INTO sessions(token_hash,user_id,created,expires) VALUES(?,?,?,?)",
           (token_hash, user_id, ts(), expires))
 return token


def public_user(row) -> dict:
 return {"id": row["id"], "email": row["email"], "name": row["name"]}


@app.post("/api/auth/register")
def register(p: Register, request: Request):
 if not limiter.allow(f"register:{client_ip(request)}", 10, 3600):
  raise HTTPException(429, "Too many sign-ups from this network. Try again later.")
 try:
  email = auth.normalize_email(p.email)
  auth.validate_password(p.password)
 except ValueError as error:
  raise HTTPException(400, str(error))
 user_id, name = str(uuid.uuid4()), (p.name or "").strip() or email.split("@")[0]
 password_hash = auth.hash_password(p.password)
 try:
  with connect() as c:
   if c.execute("SELECT 1 FROM users WHERE email=?", (email,)).fetchone():
    raise HTTPException(409, "An account with this email already exists. Try signing in.")
   c.execute("INSERT INTO users(id,email,name,password_hash,created) VALUES(?,?,?,?,?)",
             (user_id, email, name, password_hash, ts()))
   token = issue_session(c, user_id)
 except INTEGRITY_ERRORS:
  raise HTTPException(409, "An account with this email already exists. Try signing in.")
 return {"token": token, "user": {"id": user_id, "email": email, "name": name}}


@app.post("/api/auth/login")
def login(p: Login, request: Request):
 try:
  email = auth.normalize_email(p.email)
 except ValueError:
  raise HTTPException(401, "Incorrect email or password.")
 if not (limiter.allow(f"login-ip:{client_ip(request)}", 30, 600) and limiter.allow(f"login:{email}", 8, 600)):
  raise HTTPException(429, "Too many sign-in attempts. Please wait a few minutes and try again.")
 with connect() as c:
  row = c.execute("SELECT id,email,name,password_hash FROM users WHERE email=?", (email,)).fetchone()
  # Always do one password hash so response time doesn't reveal whether the email exists.
  valid = auth.verify_password(p.password, row["password_hash"] if row else auth.DUMMY_HASH)
  if not row or not valid:
   raise HTTPException(401, "Incorrect email or password.")
  token = issue_session(c, row["id"])
 return {"token": token, "user": public_user(row)}


def bearer_token(authorization: str | None) -> str:
 if authorization and authorization.lower().startswith("bearer "):
  return authorization[7:].strip()
 return ""


def current_user(authorization: str | None = Header(None)) -> dict:
 token = bearer_token(authorization)
 if not token:
  raise HTTPException(401, "Please sign in.")
 with connect() as c:
  row = c.execute(
   "SELECT u.id,u.email,u.name FROM sessions s JOIN users u ON u.id=s.user_id "
   "WHERE s.token_hash=? AND s.expires>?", (auth.hash_token(token), ts())).fetchone()
 if not row:
  raise HTTPException(401, "Your session expired. Please sign in again.")
 return public_user(row)


@app.get("/api/auth/me")
def me(user: dict = Depends(current_user)):
 return user


@app.post("/api/auth/logout")
def logout(authorization: str | None = Header(None)):
 token = bearer_token(authorization)
 if token:
  with connect() as c:
   c.execute("DELETE FROM sessions WHERE token_hash=?", (auth.hash_token(token),))
 return {"ok": True}


# --------------------------------------------------------------------------- #
# Conversations (always scoped to the signed-in user)
# --------------------------------------------------------------------------- #

class Chat(BaseModel):
 message: str = Field(min_length=1, max_length=8000)
 conversation_id: str | None = None


class NewChat(BaseModel):
 title: str = "New chat"


@app.get("/api/health")
def health():
 return {"status": "ok", "vision": vision_available()}


@app.get("/api/conversations")
def list_chats(user: dict = Depends(current_user)):
 with connect() as c:
  rows = c.execute("SELECT id,title,created,updated FROM conversations WHERE user_id=? ORDER BY updated DESC",
                   (user["id"],)).fetchall()
 return [dict(r) for r in rows]


@app.post("/api/conversations")
def new_chat(p: NewChat, user: dict = Depends(current_user)):
 cid, t = str(uuid.uuid4()), ts()
 with connect() as c:
  c.execute("INSERT INTO conversations(id,title,created,updated,user_id) VALUES(?,?,?,?,?)",
            (cid, p.title[:80], t, t, user["id"]))
 return {"id": cid, "title": p.title[:80]}


@app.get("/api/conversations/{cid}")
def get_chat(cid: str, user: dict = Depends(current_user)):
 with connect() as c:
  conv = c.execute("SELECT id,title,created,updated FROM conversations WHERE id=? AND user_id=?",
                   (cid, user["id"])).fetchone()
  if not conv:
   raise HTTPException(404, "Conversation not found")
  msgs = []
  for r in c.execute("SELECT * FROM messages WHERE conversation_id=? ORDER BY id", (cid,)).fetchall():
   d = dict(r)
   d["metadata"] = json.loads(d["metadata"] or "{}")
   msgs.append(d)
 return {"conversation": dict(conv), "messages": msgs}


@app.delete("/api/conversations/{cid}")
def delete_chat(cid: str, user: dict = Depends(current_user)):
 with connect() as c:
  if not c.execute("SELECT 1 FROM conversations WHERE id=? AND user_id=?", (cid, user["id"])).fetchone():
   raise HTTPException(404, "Conversation not found")
  c.execute("DELETE FROM messages WHERE conversation_id=?", (cid,))
  c.execute("DELETE FROM conversations WHERE id=?", (cid,))
 return {"deleted": True}


def ensure(cid: str | None, user_id: str) -> str:
 with connect() as c:
  if cid and c.execute("SELECT 1 FROM conversations WHERE id=? AND user_id=?", (cid, user_id)).fetchone():
   return cid
  cid, t = str(uuid.uuid4()), ts()
  c.execute("INSERT INTO conversations(id,title,created,updated,user_id) VALUES(?,?,?,?,?)",
            (cid, "New chat", t, t, user_id))
 return cid


def save(cid: str, role: str, text: str, meta: dict | None = None) -> None:
 t = ts()
 with connect() as c:
  c.execute("INSERT INTO messages(conversation_id,role,content,metadata,created) VALUES(?,?,?,?,?)",
            (cid, role, text, json.dumps(meta or {}), t))
  c.execute("UPDATE conversations SET updated=? WHERE id=?", (t, cid))
  if role == "user":
   r = c.execute("SELECT title FROM conversations WHERE id=?", (cid,)).fetchone()
   if r and r["title"] == "New chat":
    c.execute("UPDATE conversations SET title=? WHERE id=?", (text[:48], cid))


@app.post("/api/chat")
def chat(p: Chat, user: dict = Depends(current_user)):
 if not limiter.allow(f"chat:{user['id']}", 40, 600):
  raise HTTPException(429, "You're sending messages too quickly. Please wait a moment.")
 text = p.message.strip()
 if not text:
  raise HTTPException(400, "Empty message")
 cid = ensure(p.conversation_id, user["id"])
 save(cid, "user", text)
 with connect() as c:
  rows = c.execute("SELECT role,content FROM messages WHERE conversation_id=? ORDER BY id DESC LIMIT 12",
                   (cid,)).fetchall()
 hist = [dict(r) for r in rows][1:]
 hist.reverse()
 try:
  result = answer_math(text, hist)
 except Exception as e:  # noqa: BLE001
  logger.exception("answer_math failed")
  result = {"success": False, "operation": None, "result": None, "steps": [], "verification": None,
            "response": f"I couldn't process that request: {e}", "error": str(e)}
 save(cid, "assistant", result["response"], result)
 return {"conversation_id": cid, **result}


# --------------------------------------------------------------------------- #
# File upload: photos, handwriting, PDFs, documents -> editable question text
# --------------------------------------------------------------------------- #

@app.post("/api/extract")
def extract(files: list[UploadFile] = File(...), user: dict = Depends(current_user)):
 if not limiter.allow(f"extract:{user['id']}", 15, 600):
  raise HTTPException(429, "You're uploading too quickly. Please wait a moment.")
 if len(files) > MAX_FILES:
  raise HTTPException(400, f"Upload at most {MAX_FILES} files at a time.")
 loaded = [(f.filename or "upload", f.file.read(MAX_BYTES + 1)) for f in files]
 try:
  return extract_files(loaded)
 except ExtractError as error:
  raise HTTPException(400, str(error))
