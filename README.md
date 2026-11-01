# Mathematical AI Reasoning Agent

React chat frontend + FastAPI backend + Ollama/Qwen planner + SymPy computation and verification. SQLite stores conversation history.

## Requirements
Python 3.10+, Node.js 18+, Ollama.

## Run Ollama
```powershell
ollama pull qwen2.5:3b
```
Ensure Ollama is running (Windows app/service or `ollama serve`).

## Backend (IntelliJ terminal)
```powershell
cd backend
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
uvicorn app.main:app --reload --host 127.0.0.1 --port 8000
```
Or use your existing project `.venv2` and install `backend/requirements.txt` there. API docs: http://localhost:8000/docs

## Frontend (second terminal)
```powershell
cd frontend
npm install
npm run dev
```
Open http://localhost:5173

## Sample prompts
- Solve 2*x - 4 = 0
- Find roots of x**2 - 5*x + 6
- Derivative of x**3 + 2*x
- Integrate x**2 + 3*x
- Determinant of [[5,2],[1,3]]
- Inverse of [[1,2],[3,4]]
- Multiply [[1,2],[3,4]] by [[2,0],[1,2]]

The LLM extracts intent; SymPy computes and verifies. Fallback parsing handles common requests if Ollama is unavailable. Prototype: do not expose publicly without authentication, rate limiting, and stronger isolation.
