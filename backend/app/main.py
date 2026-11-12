import json
import logging
import os
import queue
import re
import secrets
import threading
import uuid
from datetime import datetime, timedelta, timezone
from io import BytesIO
from pathlib import Path

from dotenv import load_dotenv

# Load the project-level .env during local development; Render uses dashboard env vars.
load_dotenv(Path(__file__).resolve().parents[1] / ".env")

from fastapi import Depends, FastAPI, File, Header, HTTPException, Request, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
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


# =========================================================================== #
# Word-export math rendering (LaTeX -> OMML)
# =========================================================================== #

OMML_NS = "http://schemas.openxmlformats.org/officeDocument/2006/math"
W_NS = "http://schemas.openxmlformats.org/wordprocessingml/2006/main"

_INLINE_MATH = re.compile(r"(?<!\$)\$(?!\$)(.+?)(?<!\$)\$(?!\$)")
_DISPLAY_MATH = re.compile(r"\$\$([\s\S]+?)\$\$")


def _ensure_omml_ns(xml_str: str) -> str:
 """Make sure the fragment declares the m: and w: namespaces it uses."""
 if "xmlns:m=" in xml_str[:400]:
  return xml_str
 m = re.match(r"<([\w:]+)", xml_str)
 if not m:
  return xml_str
 return xml_str[:m.end()] + f' xmlns:m="{OMML_NS}" xmlns:w="{W_NS}"' + xml_str[m.end():]


def _latex_to_omml(latex_str: str) -> str | None:
 """Convert a LaTeX math expression to an inline <m:oMath> OMML fragment.

 Returns None when the conversion packages are not installed or the
 expression uses a construct latex2mathml cannot handle.
 """
 try:
  from latex2mathml.converter import convert as latex_to_mathml
  from mathml2omml import convert as mathml_to_omml
 except ImportError:
  return None
 try:
  mathml = latex_to_mathml(latex_str)
  omml = mathml_to_omml(mathml)
  if not (isinstance(omml, str) and omml.strip()):
   return None
  # mathml2omml may return <m:oMathPara><m:oMath>…</m:oMath></m:oMathPara>
  # for block math. We always want the inline form so it can live inside <w:p>.
  if "<m:oMathPara" in omml:
   inner = re.search(r"<m:oMath[\s>][\s\S]*?</m:oMath>", omml)
   if inner:
    omml = inner.group(0)
  return omml
 except Exception:
  logger.debug("LaTeX→OMML failed for %r", latex_str[:80], exc_info=False)
  return None


def _insert_omml(paragraph, omml_xml: str) -> bool:
 """Append an OMML fragment as a child of a python-docx paragraph."""
 from docx.oxml import parse_xml
 try:
  element = parse_xml(_ensure_omml_ns(omml_xml.strip()))
  paragraph._p.append(element)
  return True
 except Exception:
  logger.debug("OMML insert failed", exc_info=True)
  return False


_FALLBACK_SUBS = [
 (r"\\frac\{([^{}]*)\}\{([^{}]*)\}", r"(\1)/(\2)"),
 (r"\\sqrt\{([^{}]*)\}", r"√(\1)"),
 (r"\\sqrt\b", "√"),
 (r"\\pi\b", "π"), (r"\\theta\b", "θ"), (r"\\alpha\b", "α"), (r"\\beta\b", "β"),
 (r"\\gamma\b", "γ"), (r"\\delta\b", "δ"), (r"\\epsilon\b", "ε"),
 (r"\\lambda\b", "λ"), (r"\\mu\b", "μ"), (r"\\sigma\b", "σ"), (r"\\phi\b", "φ"),
 (r"\\omega\b", "ω"), (r"\\Sigma\b", "Σ"), (r"\\Delta\b", "Δ"), (r"\\Omega\b", "Ω"),
 (r"\\times\b", "×"), (r"\\div\b", "÷"), (r"\\cdot\b", "·"),
 (r"\\pm\b", "±"), (r"\\mp\b", "∓"),
 (r"\\leq\b|\\le\b", "≤"), (r"\\geq\b|\\ge\b", "≥"),
 (r"\\neq\b|\\ne\b", "≠"), (r"\\approx\b", "≈"), (r"\\equiv\b", "≡"),
 (r"\\infty\b", "∞"), (r"\\partial\b", "∂"), (r"\\nabla\b", "∇"),
 (r"\\sum\b", "∑"), (r"\\prod\b", "∏"), (r"\\int\b", "∫"),
 (r"\\in\b", "∈"), (r"\\notin\b", "∉"),
 (r"\\subset\b", "⊂"), (r"\\supset\b", "⊃"),
 (r"\\cup\b", "∪"), (r"\\cap\b", "∩"),
 (r"\\to\b|\\rightarrow\b", "→"), (r"\\Rightarrow\b|\\implies\b", "⇒"),
 (r"\\leftrightarrow\b", "↔"), (r"\\Leftrightarrow\b", "⇔"),
 (r"\\forall\b", "∀"), (r"\\exists\b", "∃"),
 (r"\\angle\b", "∠"), (r"\\perp\b", "⊥"), (r"\\parallel\b", "∥"),
 (r"\\sin\b", "sin"), (r"\\cos\b", "cos"), (r"\\tan\b", "tan"),
 (r"\\ln\b", "ln"), (r"\\log\b", "log"), (r"\\exp\b", "exp"),
 (r"\\left\b", ""), (r"\\right\b", ""),
 (r"\\[{}]", ""),
 (r"\\,", " "), (r"\\;", " "), (r"\\!", ""),
 (r"\\quad\b", "  "), (r"\\qquad\b", "    "),
]


def _latex_fallback(latex_str: str) -> str:
 """Readable plain-text fallback when OMML libraries are not installed."""
 s = latex_str
 for pattern, repl in _FALLBACK_SUBS:
  s = re.sub(pattern, repl, s)
 s = re.sub(r"\^\{([^{}]*)\}", r"^(\1)", s)
 s = re.sub(r"_\{([^{}]*)\}", r"_\1", s)
 return s.strip()


def _strip_inline_markdown(text: str) -> str:
 text = re.sub(r"\*\*(.+?)\*\*", r"\1", text)
 text = re.sub(r"(?<!\*)\*(?!\*)(.+?)(?<!\*)\*(?!\*)", r"\1", text)
 text = re.sub(r"`(.+?)`", r"\1", text)
 text = re.sub(r"\[([^\]]+)\]\([^)]+\)", r"\1", text)
 return text


def _add_text_run(paragraph, text: str) -> None:
 clean = _strip_inline_markdown(text)
 if clean:
  paragraph.add_run(clean)


def _fill_paragraph_with_math(paragraph, text: str) -> None:
 """Write `text` into `paragraph`, rendering each $…$ span as inline OMML."""
 pos = 0
 for m in _INLINE_MATH.finditer(text):
  if m.start() > pos:
   _add_text_run(paragraph, text[pos:m.start()])
  latex = m.group(1).strip()
  omml = _latex_to_omml(latex)
  if omml and _insert_omml(paragraph, omml):
   pass
  else:
   paragraph.add_run(_latex_fallback(latex))
  pos = m.end()
 if pos < len(text):
  _add_text_run(paragraph, text[pos:])


def _write_text_block(doc, block: str) -> None:
 """Write a paragraph-block (markdown + inline math) into the doc."""
 for para_block in re.split(r"\n{2,}", block):
  para_block = para_block.strip()
  if not para_block:
   continue

  head = re.match(r"^(#{1,6})\s+(.+)$", para_block)
  if head:
   level = min(len(head.group(1)) + 1, 6)
   doc.add_heading(_strip_inline_markdown(head.group(2)), level=level)
   continue

  for line in para_block.split("\n"):
   line = line.rstrip()
   if not line.strip():
    continue
   bullet = re.match(r"^[-*+]\s+(.+)$", line)
   numbered = re.match(r"^(\d+)[.)]\s+(.+)$", line)
   if bullet:
    para = doc.add_paragraph(style="List Bullet")
    _fill_paragraph_with_math(para, bullet.group(1))
   elif numbered:
    para = doc.add_paragraph(style="List Number")
    _fill_paragraph_with_math(para, numbered.group(2))
   else:
    para = doc.add_paragraph()
    _fill_paragraph_with_math(para, line)


def _write_message_body(doc, content: str) -> None:
 """Split a message into display-math blocks and text blocks, then render each."""
 pos = 0
 for m in _DISPLAY_MATH.finditer(content):
  if m.start() > pos:
   _write_text_block(doc, content[pos:m.start()])
  latex = m.group(1).strip()
  para = doc.add_paragraph()
  para.alignment = 1  # center
  omml = _latex_to_omml(latex)
  if omml and _insert_omml(para, omml):
   pass
  else:
   para.add_run(_latex_fallback(latex))
  pos = m.end()
 if pos < len(content):
  _write_text_block(doc, content[pos:])


# =========================================================================== #
# Accounts
# =========================================================================== #

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
 c.execute("DELETE FROM sessions WHERE expires<?", (ts(),))
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


# =========================================================================== #
# Conversations
# =========================================================================== #

class Chat(BaseModel):
 message: str = Field(min_length=1, max_length=8000)
 conversation_id: str | None = None
 replace_from: int | None = None


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
  c.execute("DELETE FROM shares WHERE conversation_id=?", (cid,))
  c.execute("DELETE FROM conversations WHERE id=?", (cid,))
 return {"deleted": True}


# =========================================================================== #
# Sharing
# =========================================================================== #

@app.post("/api/conversations/{cid}/share")
def share_chat(cid: str, user: dict = Depends(current_user)):
 with connect() as c:
  if not c.execute("SELECT 1 FROM conversations WHERE id=? AND user_id=?",
                   (cid, user["id"])).fetchone():
   raise HTTPException(404, "Conversation not found")
  existing = c.execute(
   "SELECT token FROM shares WHERE conversation_id=? AND user_id=?",
   (cid, user["id"])).fetchone()
  if existing:
   token = existing["token"]
  else:
   token = secrets.token_urlsafe(18)
   c.execute("INSERT INTO shares(token,conversation_id,user_id,created) VALUES(?,?,?,?)",
             (token, cid, user["id"], ts()))
 return {"token": token}


@app.delete("/api/conversations/{cid}/share")
def unshare_chat(cid: str, user: dict = Depends(current_user)):
 with connect() as c:
  if not c.execute("SELECT 1 FROM conversations WHERE id=? AND user_id=?",
                   (cid, user["id"])).fetchone():
   raise HTTPException(404, "Conversation not found")
  c.execute("DELETE FROM shares WHERE conversation_id=? AND user_id=?",
            (cid, user["id"]))
 return {"ok": True}


@app.get("/api/shared/{token}")
def get_shared(token: str):
 with connect() as c:
  share = c.execute("SELECT conversation_id FROM shares WHERE token=?",
                    (token,)).fetchone()
  if not share:
   raise HTTPException(404, "This share link is invalid or has been revoked.")
  cid = share["conversation_id"]
  conv = c.execute("SELECT id,title,created FROM conversations WHERE id=?",
                   (cid,)).fetchone()
  if not conv:
   raise HTTPException(404, "This conversation no longer exists.")
  rows = c.execute(
   "SELECT role,content,created FROM messages WHERE conversation_id=? ORDER BY id",
   (cid,)).fetchall()
 return {
  "conversation": dict(conv),
  "messages": [{"role": r["role"], "content": r["content"], "created": r["created"]}
               for r in rows],
 }


@app.get("/api/conversations/{cid}/export.docx")
def export_chat_docx(cid: str, user: dict = Depends(current_user)):
 with connect() as c:
  conv = c.execute(
   "SELECT id,title,created FROM conversations WHERE id=? AND user_id=?",
   (cid, user["id"])).fetchone()
  if not conv:
   raise HTTPException(404, "Conversation not found")
  rows = c.execute(
   "SELECT role,content FROM messages WHERE conversation_id=? ORDER BY id",
   (cid,)).fetchall()

 from docx import Document

 doc = Document()
 doc.add_heading(conv["title"] or "MathMind Chat", level=1)
 doc.add_paragraph("Exported from MathMind")

 for row in rows:
  label = "You" if row["role"] == "user" else "MathMind"
  doc.add_heading(label, level=2)
  _write_message_body(doc, row["content"] or "")

 buf = BytesIO()
 doc.save(buf)
 buf.seek(0)
 safe = re.sub(r"[^A-Za-z0-9]+", "-", conv["title"] or "chat")[:40].strip("-") or "chat"
 return StreamingResponse(
  buf,
  media_type="application/vnd.openxmlformats-officedocument.wordprocessingml.document",
  headers={"Content-Disposition": f'attachment; filename="mathmind-{safe}.docx"'},
 )


# =========================================================================== #
# Chat streaming
# =========================================================================== #

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

 if p.replace_from is not None:
  with connect() as c:
   exists = c.execute(
    "SELECT 1 FROM messages WHERE conversation_id=? AND id=?",
    (cid, p.replace_from)).fetchone()
   if not exists:
    raise HTTPException(404, "The message you tried to edit no longer exists.")
   c.execute("DELETE FROM messages WHERE conversation_id=? AND id>=?",
             (cid, p.replace_from))

 save(cid, "user", text)

 with connect() as c:
  rows = c.execute("SELECT role,content FROM messages WHERE conversation_id=? ORDER BY id DESC LIMIT 6",
                   (cid,)).fetchall()
  hist = [dict(r) for r in rows][1:]
  hist.reverse()

 q = queue.Queue()

 def run_math_agent():
  try:
   result = answer_math(text, hist, on_event=lambda t, m: q.put({"type": t, "content": m}))
   save(cid, "assistant", result["response"], result)
   q.put({"type": "result", "data": {"conversation_id": cid, **result}})
  except Exception as e:
   logger.exception("answer_math failed")
   err_result = {"success": False, "operation": None, "result": None, "steps": [],
                 "verification": None, "response": f"I couldn't process that: {e}", "error": str(e)}
   save(cid, "assistant", err_result["response"], err_result)
   q.put({"type": "result", "data": {"conversation_id": cid, **err_result}})

 def event_generator():
  thread = threading.Thread(target=run_math_agent, daemon=True)
  thread.start()

  while True:
   item = q.get()
   yield f"data: {json.dumps(item)}\n\n"
   if item["type"] == "result":
    break

 return StreamingResponse(
  event_generator(),
  media_type="text/event-stream",
  headers={"X-Accel-Buffering": "no"},
 )


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