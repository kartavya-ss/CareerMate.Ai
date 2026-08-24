# CareerMate AI

A multi-agent AI system that helps job seekers go from resume to tailored application — built with **LangGraph**, **MCP (Model Context Protocol)**, a **Supervisor agent**, **input guardrails**, and a **Human-in-the-Loop (HITL)** approval step.

Give it a resume (PDF) and a target role, and it will:
1. Parse your resume into a structured skills/experience profile
2. Search live job listings matching your target role and location
3. Compare your skills against what the roles require and flag gaps
4. Score your overall fit and identify risk areas
5. Draft a shortlist + tailored cover letter — which **you review and approve or send back for revision** before the final version is generated

---

## Why this exists

Most resume tools either just parse a PDF or just search jobs. CareerMate AI chains both together with an LLM reasoning about fit in between, and — critically — keeps a human in the loop before anything is finalized, rather than auto-generating an application end-to-end.

## Architecture

| Agent | Role |
|---|---|
| **Supervisor** | Validates the request and decides which specialist agents are needed (e.g., a "resume review only" request skips the job-search and skill-gap agents) |
| **Guardrails** | Blocks requests unrelated to job search/resume help, and attempts at prompt injection |
| **Resume Agent** | Extracts a structured profile (skills, experience, education) from the uploaded resume PDF |
| **Job Search Agent** | Uses a Tavily MCP server to pull real, current job listings matching the target role/location |
| **Skill Gap Agent** | Uses a custom local MCP server to compare resume skills against role requirements and report matched/missing skills |
| **Match Agent** | LLM reasons about overall fit, strengths, and risk areas |
| **Application Agent** | Synthesizes a shortlist + tailored cover letter into a draft |
| **Human-in-the-Loop** | Pauses the workflow for the user to approve the draft or send it back with revision feedback |
| **Final Agent** | Produces the polished, final response incorporating any human feedback |

All agents run as nodes in a single LangGraph graph with conditional routing, so the Supervisor can skip agents that aren't relevant to a given request rather than always running the full pipeline.

## Tech stack

- **LangGraph** — agent orchestration and state graph
- **MCP (Model Context Protocol)** — tool integration for both the job-search agent (Tavily) and the skill-gap agent (custom local server)
- **Groq** (`openai/gpt-oss-120b`) — LLM inference
- **FastAPI** — backend API
- **PostgreSQL** — LangGraph checkpoint persistence (enables the pause/resume HITL flow across requests)
- **pypdf** — resume PDF text extraction

## Project structure

```
app.py                          # FastAPI app + API endpoints
backend.py                      # LangGraph state, agents, routing, HITL logic
mcp_client.py                   # MCP client: Tavily + skill-gap server connections
custom_skill_gap_mcp_server.py  # Custom MCP server for skill matching
templates/index.html            # Frontend UI
static/script.js                # Frontend logic
static/style.css                # Frontend styling
```

## Setup

### Prerequisites
- Python 3.10+
- A [Groq](https://console.groq.com) API key
- A [Tavily](https://tavily.com) API key
- A Postgres database (e.g., a free [Neon](https://neon.tech) project)

### Install

```bash
python -m venv venv
venv\Scripts\activate          # Windows
pip install -r requirements.txt
```

### Configure

Create a `.env` file in the project root:

```
GROQ_API_KEY=your_groq_key
TAVILY_API_KEY=your_tavily_key
DATABASE_URL=your_postgres_connection_string
```

### Run

```bash
python app.py
```

Visit `http://127.0.0.1:8000`.

## API

- `POST /api/career` — form-data: `message` (text), `thread_id` (optional), `resume_file` (optional PDF). Starts or continues a career-assistant thread.
- `POST /api/career/approve` — JSON: `{ "thread_id": "...", "approved": true|false, "feedback": "..." }`. Resumes a paused thread after human review.
- `GET /health` — health check.

## Design decisions worth knowing about

**Why a Supervisor instead of a fixed pipeline?**
Not every request needs every agent — a plain resume review shouldn't trigger a job search. The Supervisor reads the request and picks only the relevant agents, saving LLM calls and keeping responses focused.

**Why cap intermediate context length (`_trim` helper)?**
Later agents (Match, Application, Final) build on earlier agents' output. Without limits, these prompts grow large enough to exceed provider rate limits (e.g., Groq's free-tier tokens-per-minute cap). Each agent truncates the intermediate context it receives from earlier steps to stay within limits without losing the quality of the final output.

**Why MCP instead of calling APIs directly?**
MCP decouples tool logic from agent logic — the job-search agent doesn't need to know how Tavily's API works, and the skill-gap agent's matching logic lives in its own server, independently testable and swappable.

**Why Human-in-the-Loop before finalizing?**
A generated cover letter or shortlist can be wrong, mistimed, or just not the user's voice. Pausing for explicit approval (or revision feedback) before finalizing avoids blindly sending out AI-written content on the user's behalf.

## Known limitations

- **Job listings are not guaranteed accurate or current.** The job-search agent surfaces what live web search returns, and instructs the model not to invent postings — but results can occasionally be generic or under-specific when a strong match isn't found. Always verify a listing on the company's site before applying.
- **Skill-gap matching uses a small built-in skill taxonomy** for a handful of common roles (backend/frontend developer, data analyst, data scientist) plus keywords found in retrieved job listings. It is not a comprehensive skills database.
- **No automated test suite yet** — the project has been manually verified end-to-end (resume upload → draft → approve/revise → final output) but does not yet have unit tests.

## Acknowledgements

Built on top of the architecture pattern from an open-source LangGraph + MCP + Supervisor + Guardrails + HITL demo project, adapted here into a job-application assistant with a new agent set, a new custom MCP server, PDF resume upload, and rate-limit-aware prompt design.

## License

See `LICENSE`.