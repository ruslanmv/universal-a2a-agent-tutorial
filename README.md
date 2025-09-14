# Build a Production‑Ready Universal A2A Agent — with IBM watsonx.ai, MatrixHub & MCP Gateway *(Pro Edition)*

> This end‑to‑end tutorial upgrades the original guide with rock‑solid steps to publish an A2A agent to **MatrixHub** and register it in **MCP Gateway**. It keeps everything you had and **extends it** with production‑safe ingestion flows, manifest templates, and copy‑paste scripts. The result scales, stays vendor‑neutral, and remains easy to evolve.

---

## What you’ll build

A single HTTP service (FastAPI) that exposes:

* `/a2a` (**raw A2A**)
* `/rpc` (**JSON‑RPC 2.0**)
* `/openai/v1/chat/completions` (**OpenAI‑compatible**)
* `/.well-known/agent-card.json` for discovery
* `/healthz` & `/readyz` for ops

You’ll wire:

* A **pluggable Provider** (IBM **watsonx.ai** in this tutorial; swap via env)
* A **pluggable Framework** layer (LangGraph, CrewAI, LangChain)
* A **publishable agent** for **MatrixHub** + optional **MCP Gateway** registration

---

## Prerequisites

* Python 3.11+
* `git`, `make`
* (optional) Docker / Docker Compose
* IBM **watsonx.ai** credentials:

  * `WATSONX_API_KEY`
  * `WATSONX_URL` (e.g., `https://us-south.ml.cloud.ibm.com`)
  * `WATSONX_PROJECT_ID`
  * *(optional)* `MODEL_ID` (defaults to `ibm/granite-3-3-8b-instruct` in the sample provider)
* *(Optional for later sections)* A running **MatrixHub** and **MCP Gateway** instance

> **What are MatrixHub & MCP Gateway?**
>
> * **MatrixHub** is a service catalog for discovering and installing agents, tools, and servers. It stores A2A manifests and can optionally register your agents in an MCP Gateway.
> * **MCP Gateway** acts as a secure entry point for clients to talk to registered agents over standardized transports.

---

## 1) Clone & install

```bash
git clone https://github.com/ruslanmv/universal-a2a-agent.git
cd universal-a2a-agent

python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate

pip install -e .
# Adapters used in this tutorial
pip install -e .[langgraph]
pip install -e .[langchain]
pip install -e .[crewai]
```

> Tip: `pip install -e .[all]` installs every adapter.

---

## 2) Getting your watsonx.ai credentials (quick onboarding)

1. Sign up or sign in to **IBM Cloud**.
2. Create (or open) a **watsonx.ai** project.
3. In the project, create a **service API key**.
4. Collect:

   * `WATSONX_API_KEY` — your API key
   * `WATSONX_URL` — region endpoint, e.g. `https://us-south.ml.cloud.ibm.com`
   * `WATSONX_PROJECT_ID` — the GUID of your project
5. *(Optional)* **Models**: you can use the default `ibm/granite-3-3-8b-instruct` or choose another foundation model available to your account. Set `MODEL_ID` accordingly.

> You can also store these in a `.env` file at the project root:
>
> ```env
> LLM_PROVIDER=watsonx
> WATSONX_API_KEY=your_api_key
> WATSONX_URL=https://us-south.ml.cloud.ibm.com
> WATSONX_PROJECT_ID=your_project_id
> MODEL_ID=ibm/granite-3-3-8b-instruct
> AGENT_FRAMEWORK=langgraph
> PUBLIC_URL=http://localhost:8000
> ```

---

## 3) Configure the Provider & Framework

Pick **watsonx.ai** as Provider and a Framework (we’ll start with LangGraph):

```bash
export LLM_PROVIDER=watsonx
export WATSONX_API_KEY=YOUR_KEY
export WATSONX_URL=https://us-south.ml.cloud.ibm.com
export WATSONX_PROJECT_ID=YOUR_PROJECT_ID
# optional
# export MODEL_ID=ibm/granite-3-3-8b-instruct

export AGENT_FRAMEWORK=langgraph   # or: crewai, langchain
```

---

## 4) Run the server

```bash
make run
# or
uvicorn a2a_universal.server:app --host 0.0.0.0 --port 8000
```

Smoke‑test:

```bash
open http://localhost:8000/docs
curl -s http://localhost:8000/healthz
curl -s http://localhost:8000/readyz | jq
curl -s http://localhost:8000/.well-known/agent-card.json | jq
```

---

## 5) Talk to the agent — three ways

### A) Raw A2A

```bash
curl -s http://localhost:8000/a2a -H 'Content-Type: application/json' -d '{
  "method":"message/send",
  "params":{"message":{
    "role":"user","messageId":"m1",
    "parts":[{"type":"text","text":"ping from A2A"}]
  }}
}' | jq
```

### B) JSON‑RPC 2.0

```bash
curl -s http://localhost:8000/rpc -H 'Content-Type: application/json' -d '{
  "jsonrpc":"2.0","id":"1","method":"message/send",
  "params":{"message":{
    "role":"user","messageId":"cli",
    "parts":[{"type":"text","text":"hello via jsonrpc"}]
  }}
}' | jq
```

### C) OpenAI‑compatible

```bash
curl -s http://localhost:8000/openai/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{
    "model":"universal-a2a-hello",
    "messages":[{"role":"user","content":"hello from openai route"}]
  }' | jq -r '.choices[0].message.content'
```

---

## 6) Use from frameworks (A2A → watsonx.ai)

> **Note on orchestration LLMs**: Some frameworks (LangChain, CrewAI) may require an LLM for the planner/orchestrator itself (e.g., `ChatOpenAI`). This is separate from the actual **work**, which we route through A2A → watsonx. If needed, set an API key for the framework’s LLM (e.g., `OPENAI_API_KEY`).

### 6.1 LangChain Tool

```python
# examples/quickstart_langchain_watsonx.py
import httpx
from langchain.agents import initialize_agent, AgentType
from langchain_core.tools import Tool
from langchain_openai import ChatOpenAI

BASE = "http://localhost:8000"

def a2a_call(prompt: str) -> str:
    payload = {
        "method": "message/send",
        "params": {"message": {
            "role": "user", "messageId": "lc-tool",
            "parts": [{"type": "text", "text": prompt}],
        }},
    }
    r = httpx.post(f"{BASE}/a2a", json=payload, timeout=30.0)
    r.raise_for_status()
    data = r.json()
    for p in (data.get("message") or {}).get("parts", []):
        if p.get("type") == "text":
            return p.get("text", "")
    return ""

tool = Tool(name="a2a_hello", description="Send a prompt to the Universal A2A agent.", func=a2a_call)
llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
agent = initialize_agent([tool], llm, agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION)

if __name__ == "__main__":
    print(agent.run("Use the a2a_hello tool to say hello to LangChain."))
```

### 6.2 LangGraph Node

```python
# examples/quickstart_langgraph_watsonx.py
import asyncio, httpx
from langgraph.graph import StateGraph, END, MessagesState
from langchain_core.messages import HumanMessage, AIMessage

BASE = "http://localhost:8000"

async def a2a_send(text: str) -> str:
    payload = {
        "method": "message/send",
        "params": {"message": {
            "role": "user", "messageId": "lg-node",
            "parts": [{"type": "text", "text": text}],
        }},
    }
    async with httpx.AsyncClient(timeout=30.0) as client:
        r = await client.post(f"{BASE}/a2a", json=payload)
        r.raise_for_status()
        data = r.json()
        for p in (data.get("message") or {}).get("parts", []):
            if p.get("type") == "text":
                return p.get("text", "")
    return ""

async def a2a_node(state: dict) -> dict:
    last = state["messages"][-1]
    reply = await a2a_send(getattr(last, "content", ""))
    return {"messages": [AIMessage(content=reply)]}

g = StateGraph(MessagesState)
g.add_node("a2a", a2a_node)
g.add_edge("__start__", "a2a")
g.add_edge("a2a", END)
app = g.compile()

async def main():
    out = await app.ainvoke({"messages": [HumanMessage(content="ping")]})
    print(out["messages"][-1].content)

if __name__ == "__main__":
    asyncio.run(main())
```

### 6.3 CrewAI — **Duo** (Researcher → Writer) with watsonx via A2A *(Extended)*

> Save as `examples/crewai_watsonx_duo.py`

```python
import os
import httpx
from crewai import Agent, Task, Crew

BASE = os.getenv("A2A_BASE", "http://localhost:8000")

def a2a_call(prompt: str) -> str:
    payload = {
        "method": "message/send",
        "params": {
            "message": {
                "role": "user",
                "messageId": "crewai-duo",
                "parts": [{"type": "text", "text": prompt}],
            }
        },
    }
    r = httpx.post(f"{BASE}/a2a", json=payload, timeout=30.0)
    r.raise_for_status()
    data = r.json()
    for p in (data.get("message") or {}).get("parts", []):
        if p.get("type") == "text":
            return p.get("text", "")
    return ""

if __name__ == "__main__":
    topic = "Edge AI for autonomous drones in search & rescue"

    researcher = Agent(
        role="Researcher",
        goal="Gather concise, accurate notes and outline the topic.",
        backstory="Methodical analyst who drafts clean bullet-point notes.",
        tools=[a2a_call],
        allow_delegation=False,
        verbose=True,
    )

    writer = Agent(
        role="Writer",
        goal="Turn notes into a tidy LaTeX article (1–2 pages).",
        backstory="Technical writer who produces compilable LaTeX.",
        tools=[a2a_call],
        allow_delegation=False,
        verbose=True,
    )

    t_research = Task(
        description=(
            f"Research the topic: '{topic}'. "
            "Use the a2a_call tool to produce a concise outline with bullet points, "
            "covering background, key challenges, approaches, and example applications. "
            "Output: a Markdown outline."
        ),
        agent=researcher,
        expected_output="A clean Markdown outline of findings.",
    )

    t_write = Task(
        description=(
            "Using the outline from the Researcher, write a compilable LaTeX article. "
            "Use the a2a_call tool to help with prose and LaTeX formatting. "
            "Return only the final .tex content."
        ),
        agent=writer,
        context=[t_research],
        expected_output="A single LaTeX .tex string, compilable.",
    )

    crew = Crew(agents=[researcher, writer], tasks=[t_research, t_write])
    result = crew.kickoff()
    print("\n=== FINAL LATEX ===\n")
    print(result)
```

> **Enhancement ideas**: Add a **Reviewer** agent for quality and LaTeX fixes, and persist the final `.tex` to disk.

**Reviewer add‑on (tiny diff):**

```python
# Add after writer definition
reviewer = Agent(
    role="Reviewer",
    goal="Improve clarity and fix LaTeX issues without changing intent.",
    backstory="Meticulous editor who ensures the .tex compiles.",
    tools=[a2a_call],
    allow_delegation=False,
    verbose=True,
)

# New task that depends on writer output
t_review = Task(
    description=(
        "Review the LaTeX from the Writer for clarity and LaTeX correctness. "
        "Use the a2a_call tool for edits if needed. "
        "Return the corrected final .tex only."
    ),
    agent=reviewer,
    context=[t_write],
    expected_output="Corrected final .tex (single string).",
)

crew = Crew(agents=[researcher, writer, reviewer], tasks=[t_research, t_write, t_review])
```

> Optional: write to file
>
> ```python
> with open("final.tex", "w", encoding="utf-8") as f:
>     f.write(result)
> print("Saved to final.tex")
> ```

---

## 7) Containerize & run in Docker

**Build & run locally:**

```bash
docker build -t yourrepo/universal-a2a-agent:1.2.0 .
docker run --rm -p 8000:8000 \
  -e PUBLIC_URL=http://localhost:8000 \
  -e LLM_PROVIDER=watsonx \
  -e WATSONX_API_KEY=... \
  -e WATSONX_URL=... \
  -e WATSONX_PROJECT_ID=... \
  yourrepo/universal-a2a-agent:1.2.0
```

> Replace `yourrepo` with your Docker Hub username or your registry path.

**Compose (optional):**

```yaml
version: "3.9"
services:
  a2a-agent:
    image: yourrepo/universal-a2a-agent:1.2.0
    ports: ["8000:8000"]
    environment:
      PUBLIC_URL: http://localhost:8000
      LLM_PROVIDER: watsonx
      WATSONX_API_KEY: ${WATSONX_API_KEY}
      WATSONX_URL: ${WATSONX_URL}
      WATSONX_PROJECT_ID: ${WATSONX_PROJECT_ID}
    restart: unless-stopped
```

---

## 8) Publish to MatrixHub (A2A‑ready)

MatrixHub stores A2A details under `entity.manifests.a2a` and tags `entity.protocols += ["a2a@<version>"]` for discovery.

### 8.1 Author your catalog manifest — `agent.manifest.yaml`

```yaml
schema_version: 1

type: agent
id: universal-a2a-hello
version: 1.2.0
name: Universal A2A — Hello
summary: JSON‑RPC echo agent, backed by IBM watsonx.ai
homepage: https://example.com/universal-a2a
license: Apache-2.0
capabilities: ["echo", "summarize"]

manifests:
  a2a:
    version: "1.0"
    endpoint_url: "https://your-agent.example.com/a2a"  # or /rpc
    agent_type: jsonrpc
    auth: { type: none, value: "" }  # or bearer/api_key
    tags: ["watsonx", "demo"]
    server:
      name: universal-a2a-hello-server
      description: Gateway virtual server exposing this A2A agent
```

### 8.2 Publish an index — `index.json`

Any supported shape is fine (MatrixHub is permissive):

```json
{ "manifests": ["https://your-host/agent.manifest.yaml"] }
```

or

```json
{ "items": [{"manifest_url": "https://your-host/agent.manifest.yaml"}] }
```

or

```json
{ "entries": [{"path": "agent.manifest.yaml", "base_url": "https://your-host/"}] }
```

### 8.3 Ingest into MatrixHub

```bash
export HUB_BASE=${HUB_BASE:-http://localhost:443}

curl -s -X POST "$HUB_BASE/catalog/ingest" \
  -H 'Content-Type: application/json' \
  -d '{ "index_url": "https://your-host/index.json" }' | jq
```

**Expected:** entity created; `entity.manifests.a2a` present; `entity.protocols` includes `a2a@1.0`.

### 8.4 Install the agent (and best‑effort register to Gateway)

```bash
curl -s -X POST "$HUB_BASE/catalog/install" \
  -H 'Content-Type: application/json' \
  -d '{ "id": "agent:universal-a2a-hello@1.2.0", "target": "/tmp/myapp" }' | jq
```

> If your A2A manifest includes a `server` block, MatrixHub will try to create a virtual server in the Gateway and associate this agent. It won’t fail the install if the Gateway is unreachable.

---

## 9) Register directly with MCP Gateway (optional)

A tiny helper client is enough. Example shape:

```python
# scripts/register_a2a.py
from typing import Optional
import os, requests

GATEWAY = os.getenv("MCP_GATEWAY_URL", "http://localhost:4444")
TOKEN: Optional[str] = os.getenv("MCP_GATEWAY_TOKEN")  # optional Bearer token

HEADERS = {"Content-Type": "application/json"}
if TOKEN:
    HEADERS["Authorization"] = TOKEN

agent = {
    "name": "universal-a2a-hello",
    "endpoint_url": "https://your-agent.example.com/a2a",
    "agent_type": "jsonrpc",
    "auth_type": "none",
    "auth_value": None,
    "tags": ["watsonx", "demo"],
}

# 1) Register agent (idempotent suggestion)
resp = requests.post(f"{GATEWAY}/agents/a2a", json=agent, headers=HEADERS, timeout=15)
print("register a2a:", resp.status_code, resp.text)

# 2) Create a virtual server referencing the agent
server = {
    "name": "universal-a2a-hello-server",
    "description": "Gateway virtual server exposing the A2A agent",
    "associated_a2a_agents": [agent["name"]],
}
resp = requests.post(f"{GATEWAY}/servers", json=server, headers=HEADERS, timeout=15)
print("create server:", resp.status_code, resp.text)
```

Run:

```bash
export MCP_GATEWAY_URL="https://gateway.example.com"
# optional: export MCP_GATEWAY_TOKEN="Bearer <token>"
python scripts/register_a2a.py
```

---

## 10) Call your **deployed** agent from a watsonx.ai notebook

Once your agent is reachable over HTTPS (via Docker, Compose, or Kubernetes ingress), you can call the **OpenAI‑compatible** route from any notebook:

```python
import os, requests

BASE = os.getenv("A2A_BASE", "https://your-host")
TOKEN = os.getenv("A2A_TOKEN", "")  # optional

headers = {"Content-Type": "application/json"}
if TOKEN:
    headers["Authorization"] = f"Bearer {TOKEN}"

body = {
    "model": "universal-a2a-hello",
    "messages": [{"role": "user", "content": "ping from watsonx.ai"}],
}

r = requests.post(f"{BASE}/openai/v1/chat/completions", json=body, headers=headers, timeout=20)
r.raise_for_status()
print(r.json()["choices"][0]["message"]["content"])
```

---

## 11) Golden templates (prompts & ops)

**Crew/LLM task**

```
You are a {role}. Your goal:
{goal}

Constraints:
- Be concise and correct.
- For generation/summarization, call the external A2A tool.

Deliverable:
- {expected_output}
```

**System prompt**

```
You route generation to a Universal A2A backend.
- Never leak provider credentials.
- Prefer short, actionable outputs.
- If input is noisy, clarify assumptions first.
```

**Ops checklist**

* HTTPS/TLS everywhere; set `PUBLIC_URL` accordingly
* Protect `/openai/v1/chat/completions` with auth (bearer/API key)
* Add `/healthz` + `/readyz` probes
* Structured JSON logs; centralize
* Rate-limit public ingress
* Pin images by digest in prod

---

## 12) Troubleshooting

* **`/readyz` shows not ready** → check credentials and env (`curl -s /readyz | jq` gives reasons).
* **Framework examples error** → some frameworks need a planner LLM; set `OPENAI_API_KEY` (or your alternative) for the *framework*, not for A2A.
* **Gateway 401/403** → set `MCP_GATEWAY_TOKEN` or enable JWT minting.
* **MatrixHub ingest skips A2A** → ensure your DB schema includes `entity.protocols` and `entity.manifests` (JSONB). Ingest will still work; A2A won’t persist without schema.
* **CORS / Browser calls** → enable CORS in the server env & middleware.

---

## 13) Why this design scales

* **Protocol‑first**: Agent Card + A2A manifest decouple your app from frameworks
* **Safe ingestion**: MatrixHub stores A2A blocks & tags protocols; idempotent, concurrent fetch, single‑threaded DB writes
* **Gateway client**: HTTP/2 pooled clients, retries, idempotency on Conflict (pattern)
* **Swappable providers**: single env var away (`LLM_PROVIDER`)
* **Zero lock‑in**: your application speaks HTTP to a single facade

> Ship agentic apps faster — with one A2A facade, production‑grade plumbing, and frictionless publishing to MatrixHub + MCP Gateway.
