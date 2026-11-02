# Polycrit — Evaluation-First Multi-Agent Code Reviewer

> Paste code or a GitHub repo URL. A **panel of AI agents** reviews it — architecture, security, performance, test gaps, style — then a **Critic** re-checks every finding against the actual code and removes the ones the agents made up, and a **Moderator** writes the final report, including an honest *"What I could NOT verify"* section.

![stack](https://img.shields.io/badge/backend-FastAPI-5eead4) ![llm](https://img.shields.io/badge/LLM-Groq%20LLaMA%203.3%20(free)-a78bfa) ![agents](https://img.shields.io/badge/agents-5%20reviewers%20%2B%20critic%20%2B%20moderator-60a5fa)

## Why I built this

Most AI portfolios stop at a single "RAG chatbot." Polycrit goes further: a **multi-agent system with a self-correcting critic** — the exact skill set 2026 AI/backend roles hire for (agentic orchestration, retrieval, evaluation, honest reporting). The Critic is the differentiator: where other reviewers dump whatever an LLM says, Polycrit verifies each finding against the code and deletes hallucinations. That evaluation-first mindset comes from my earlier PromptLab project (systematic evaluation of LLM outputs).

## How it works

```
GitHub URL ──┐
             ├──▶ Ingest (clone + chunk) ──▶ Retriever (TF-IDF over code)
Pasted code ─┘                                      │
                                                    ▼
   ┌───────────────────────────────────────────────────────────────┐
   │  REVIEWER PANEL (each retrieves the code it needs)             │
   │   1. Architecture   2. Security   3. Performance                │
   │   4. Test-Gap       5. Style                                  │
   │         │  (structured findings: title, severity, file, fix)   │
   └─────────┼─────────────────────────────────────────────────────-┘
             ▼
   ┌───────────────────────────────────────────────────────────────┐
   │  CRITIC — re-checks each finding against the actual code        │
   │  keep · downgrade · remove  (removes hallucinated findings)     │
   └─────────┬─────────────────────────────────────────────────────-┘
             ▼
   ┌───────────────────────────────────────────────────────────────┐
   │  MODERATOR — synthesizes the final markdown report              │
   │  Summary · Architecture · Security · Performance · Test Gaps   │
   │  · Style · Top Recommendations · What I could NOT verify        │
   └────────────────────────────────────────────────────────────────┘
```

Everything is **streamed live** to the browser via SSE — you watch each agent start, finish, and emit findings as it works.

## Tech stack

| Layer | Tech |
|---|---|
| API | FastAPI (Python), SSE streaming |
| Agent orchestration | a small, explicit graph (`backend/agents.py`) — no black-box framework, so the logic is readable and defensible in an interview |
| LLM | configurable; defaults to Groq's free LLaMA 3.3 70B (OpenAI-compatible, no paid key) |
| Retrieval | scikit-learn TF-IDF nearest-neighbour over code chunks (offline, no embedding API) |
| Frontend | single-page HTML/CSS/JS served by FastAPI |

## Quick start (local)

```bash
cd polycrit
python -m venv .venv
# Windows:
.venv\Scripts\activate
# macOS/Linux:
source .venv/bin/activate

pip install -r backend/requirements.txt

cp .env.example .env          # add your free Groq key (https://console.groq.com/keys)

cd backend
uvicorn main:app --reload
# open http://localhost:8000
```

Paste a public repo URL (e.g. a small Python project) or paste code, pick the language, hit **Run Review**.

## API

| Method | Route | Returns |
|---|---|---|
| GET | `/` | Web UI |
| GET | `/api/health` | `{status:"ok"}` |
| POST | `/api/review` | SSE stream of events: `start` → `agent_start`/`agent_done` (×5) → `critic_start`/`critic_done` → `moderator_start` → `report` → `done` |

## What's honest about it (recruiters read this part)

- The orchestrator is hand-written, so you can trace exactly how state flows between agents — no framework magic.
- The **Critic** actually removes findings it can't verify against the code; the report's "What I could NOT verify" section is generated, not bolted on.
- Retrieval uses TF-IDF, not a paid embedding model, so it runs offline after install. Trade-off: weaker semantic matches than embeddings (see upgrades).
- Confidence is per-finding; severities can be downgraded by the Critic.

## Deploy for free (no credit card)

Polycrit serves its own frontend, so it's **one deploy** — no separate frontend host needed.

### Option A — Koyeb (recommended, git-driven, free tier)
1. Push this repo to GitHub.
2. Sign up at https://www.koyeb.com with GitHub (no credit card).
3. **Create Service → GitHub → select this repo**.
4. Builder: Python. Build command: `pip install -r backend/requirements.txt`. Run command: `uvicorn main:app --host 0.0.0.0 --port $PORT --app-dir backend`.
5. Add environment variable `LLM_API_KEY` (your Groq key) and `LLM_MODEL=llama-3.3-70b-versatile`.
6. Deploy. You get `https://<your-service>.koyeb.app`.

### Option B — Render (free web service, sleeps when idle)
1. Push this repo to GitHub.
2. https://render.com → **New → Web Service → connect repo**.
3. Runtime: Python 3. Build: `pip install -r backend/requirements.txt`. Start: `uvicorn main:app --host 0.0.0.0 --port $PORT --app-dir backend`.
4. Add environment variable `LLM_API_KEY`. Deploy.

### Option C — split frontend (Vercel) + backend (Render/Koyeb)
For a faster UI: put `frontend/index.html` on Vercel/Netlify/GitHub Pages, point its `fetch` to the backend URL, and enable CORS on the backend. One-host (A/B) is simpler and recommended.

## Planned upgrades (good first issues)

- Swap TF-IDF for **ChromaDB** embeddings for semantic code retrieval.
- Port the orchestrator to **LangGraph** for conditional routing + checkpointing.
- **Human-in-the-loop**: let the user dismiss a finding and re-run the Moderator.
- Per-file **diff review** (review a git diff, not the whole repo).
- Caching of retriever index across runs.

## Project structure

```
polycrit/
├── backend/
│   ├── main.py          # FastAPI app, SSE streaming endpoint, serves UI
│   ├── agents.py        # the multi-agent graph: 5 reviewers + critic + moderator
│   ├── retrieval.py     # load/clone, chunk, TF-IDF retriever
│   ├── llm.py           # configurable OpenAI-compatible client (Groq default) + JSON parse
│   └── requirements.txt
├── frontend/
│   └── index.html       # live streaming UI (inline CSS/JS, no build step)
├── Procfile
├── .env.example
└── README.md
```

## License

MIT — see [LICENSE](LICENSE). Built from scratch by **Yaswanth Kumar Abotula**. The multi-agent patterns were informed by studying open agentic-review demos; all code here is original.
