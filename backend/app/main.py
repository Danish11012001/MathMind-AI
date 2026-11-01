import json,sqlite3,uuid
from pathlib import Path
from contextlib import closing
from datetime import datetime,timezone
from fastapi import FastAPI,HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel,Field
from .math_agent import answer_math
DB=Path(__file__).resolve().parent.parent/"history.db"
app=FastAPI(title="Mathematical AI Reasoning Agent")
app.add_middleware(CORSMiddleware,allow_origins=["http://localhost:5173"],allow_methods=["*"],allow_headers=["*"])
def ts(): return datetime.now(timezone.utc).isoformat()
def db():
 c=sqlite3.connect(DB);c.row_factory=sqlite3.Row;return c
@app.on_event("startup")
def init():
 with closing(db()) as c:
  c.executescript("CREATE TABLE IF NOT EXISTS conversations(id TEXT PRIMARY KEY,title TEXT,created TEXT,updated TEXT); CREATE TABLE IF NOT EXISTS messages(id INTEGER PRIMARY KEY,conversation_id TEXT,role TEXT,content TEXT,metadata TEXT,created TEXT);");c.commit()
class Chat(BaseModel):
 message:str=Field(min_length=1,max_length=5000)
 conversation_id:str|None=None
class NewChat(BaseModel): title:str="New chat"
@app.get("/api/health")
def health(): return {"status":"ok"}
@app.get("/api/conversations")
def list_chats():
 with closing(db()) as c:return [dict(r) for r in c.execute("SELECT * FROM conversations ORDER BY updated DESC")]
@app.post("/api/conversations")
def new_chat(p:NewChat):
 cid=str(uuid.uuid4());t=ts()
 with closing(db()) as c:c.execute("INSERT INTO conversations VALUES(?,?,?,?)",(cid,p.title[:80],t,t));c.commit()
 return {"id":cid,"title":p.title[:80]}
@app.get("/api/conversations/{cid}")
def get_chat(cid):
 with closing(db()) as c:
  conv=c.execute("SELECT * FROM conversations WHERE id=?",(cid,)).fetchone()
  if not conv:raise HTTPException(404,"Conversation not found")
  msgs=[]
  for r in c.execute("SELECT * FROM messages WHERE conversation_id=? ORDER BY id",(cid,)):
   d=dict(r);d["metadata"]=json.loads(d["metadata"] or "{}");msgs.append(d)
  return {"conversation":dict(conv),"messages":msgs}
@app.delete("/api/conversations/{cid}")
def delete_chat(cid):
 with closing(db()) as c:
  c.execute("DELETE FROM messages WHERE conversation_id=?",(cid,));n=c.execute("DELETE FROM conversations WHERE id=?",(cid,)).rowcount;c.commit()
  if not n:raise HTTPException(404,"Conversation not found")
 return {"deleted":True}
def ensure(cid):
 with closing(db()) as c:
  if cid and c.execute("SELECT id FROM conversations WHERE id=?",(cid,)).fetchone():return cid
  cid=str(uuid.uuid4());t=ts();c.execute("INSERT INTO conversations VALUES(?,?,?,?)",(cid,"New chat",t,t));c.commit();return cid
def save(cid,role,text,meta=None):
 t=ts()
 with closing(db()) as c:
  c.execute("INSERT INTO messages(conversation_id,role,content,metadata,created) VALUES(?,?,?,?,?)",(cid,role,text,json.dumps(meta or {}),t))
  c.execute("UPDATE conversations SET updated=? WHERE id=?",(t,cid))
  if role=="user":
   r=c.execute("SELECT title FROM conversations WHERE id=?",(cid,)).fetchone()
   if r["title"]=="New chat":c.execute("UPDATE conversations SET title=? WHERE id=?",(text[:48],cid))
  c.commit()
@app.post("/api/chat")
def chat(p:Chat):
 text=p.message.strip()
 if not text:raise HTTPException(400,"Empty message")
 cid=ensure(p.conversation_id);save(cid,"user",text)
 with closing(db()) as c:hist=[dict(r) for r in c.execute("SELECT role,content FROM messages WHERE conversation_id=? ORDER BY id DESC LIMIT 12",(cid,)).fetchall()][1:]
 hist.reverse()
 try:result=answer_math(text,hist)
 except Exception as e:result={"success":False,"operation":None,"result":None,"steps":[],"verification":None,"response":f"I couldn't process that request: {e}","error":str(e)}
 save(cid,"assistant",result["response"],result)
 return {"conversation_id":cid,**result}
