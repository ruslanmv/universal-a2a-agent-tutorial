# Building a Production-Ready, Pluggable A2A Agent with IBM watsonx.ai, MatrixHub, and MCP Gateway

This comprehensive, professional-grade tutorial guides you through the process of architecting, building, and deploying a vendor-neutral **Agent-to-Agent (A2A)** service. We will upgrade the foundational concepts of the Universal A2A Agent to a production-ready state by integrating it with **IBM watsonx.ai**, publishing it to the **MatrixHub** service catalog, and registering it with the **MCP Gateway** for secure, standardized communication.

The core architectural principle is **decoupling**: the agent's business logic is separated from the underlying Large Language Model (LLM) provider and the high-level orchestration framework. This design ensures that your agent is extensible, maintainable, and not locked into any single technology stack.

-----

## Architectural Overview

You will run a single, containerized **FastAPI** service that acts as a universal A2A hub. The service exposes a small, stable protocol surface so **any client, framework, or gateway can integrate without SDK lock‑in**. MatrixHub provides catalog/discovery, and MCP Gateway provides secure, standardized ingress/routing.

---

## Core Components (at a glance, no tables)

* **Agent Service (FastAPI):** The core HTTP application that implements A2A behaviors and adapters. Designed to sit behind TLS and ship with health probes and discovery.
* **Protocol Surface:** A compact set of interoperable endpoints (see *Endpoints* below) so different clients can connect without rewriting.
* **Framework Adapters (Pluggable):** Orchestration glue for **LangGraph**, **CrewAI**, **LangChain**, etc. Swap at runtime via `AGENT_FRAMEWORK`.
* **Provider (Pluggable):** The LLM/backend implementation. This guide uses **IBM watsonx.ai**; swap to OpenAI, Ollama, Anthropic, Gemini, Bedrock, etc., via `LLM_PROVIDER` without code changes.
* **MatrixHub (Catalog):** Discovery/search/install layer. Ingests your manifest, persists `manifests.a2a`, and tags `protocols=["a2a@<ver>"]` so A2A capability is queryable.
* **MCP Gateway (Ingress):** Secure entry point for clients. Supports **`POST /a2a`** agent registration and optional **virtual servers** via **`POST /servers`** that route to your A2A agent.

---

## Endpoints (data plane)

* `POST /a2a` — **Raw A2A** request/response (vendor‑neutral, lowest level).
* `POST /rpc` — **JSON‑RPC 2.0** wrapper for the same A2A methods (e.g., `message/send`).
* `POST /openai/v1/chat/completions` — **OpenAI‑compatible** shim for immediate ecosystem compatibility.
* `GET /.well-known/agent-card.json` — **Agent Card** for discovery and metadata.
* `GET /healthz` and `GET /readyz` — Liveness & readiness (the latter returns actionable reasons).

> **Auth tip:** If the agent is protected, prefer **Bearer** or **API key**; declare this in your A2A manifest under `manifests.a2a.auth` so clients and gateways know how to call you.

---

## How the pieces fit

### Data plane (runtime calls)

```
Client / App / Framework  ──HTTP──>  Universal A2A Agent  ──SDK/API──>  Provider (watsonx.ai)
```

* Your app never links vendor SDKs; it speaks **HTTP** to the A2A service.
* The **Provider** is injected via `LLM_PROVIDER` (e.g., `watsonx`, `openai`, `ollama`).
* The **Framework adapter** (LangGraph/CrewAI/LangChain) orchestrates on top of the same A2A core.

### Control & discovery plane

```
Author manifest (agent.manifest.yaml)
        │
        ▼
MatrixHub — Ingest (persists manifests.a2a, tags protocols)
        │
        ▼
MatrixHub — Install (best‑effort register to MCP Gateway /a2a; optional /servers)
        │
        ▼
MCP Gateway — Clients call Gateway; Gateway routes to your Agent
```

* **Ingest**: Makes the agent discoverable and stores A2A metadata (`manifests.a2a` + `protocols`).
* **Install**: If `MCP_GATEWAY_URL` is configured, the Hub will best‑effort POST to **`/a2a`** on the Gateway and, if your manifest includes `manifests.a2a.server`, create a **virtual server** via **`/servers`**.

---

## Mental model (one slide)

```
            ┌──────────────────────────────────────┐
            │            MatrixHub (Catalog)       │
            │  • Search, discovery, install       │
            │  • Stores manifests.a2a + protocols │
            └───────────────┬──────────────────────┘
                            │ (install triggers best‑effort registration)
                            ▼
┌───────────────────────────────────────────────────────────────────────┐
│                         MCP Gateway (Ingress)                         │
│  • /a2a registration, optional /servers                              │
│  • Auth, routing, policy                                             │
└───────────────┬───────────────────────────────────────────────────────┘
                │ (HTTPS)
                ▼
         ┌──────────────────────────┐
         │  Universal A2A Agent     │  ◄──  /.well-known/agent-card.json
         │  • /a2a  /rpc  /openai   │
         │  • /healthz  /readyz     │
         └──────────┬───────────────┘
                    │ (SDK/API)
                    ▼
            Provider (watsonx.ai)
```

---

## Portability & upgrades (why this scales)

* **Swap providers** without code changes: `LLM_PROVIDER=watsonx|openai|ollama|…`.
* **Swap orchestrators**: `AGENT_FRAMEWORK=langgraph|crewai|langchain|…`.
* **Stable protocol surface** keeps clients unchanged as internals evolve.
* **MatrixHub A2A‑ready** ingestion stores protocol‑native blocks for future tooling.
* **Gateway idempotency**: treat `/a2a` and `/servers` as idempotent during installs (409 ⇒ OK when requested).

---

## Quick responsibilities checklist

* **Agent:** implement endpoints; emit clear readiness reasons; return structured A2A responses.
* **Manifest:** include `manifests.a2a` (version, endpoint\_url, agent\_type, auth, tags; optional `server`).
* **MatrixHub:** ingest manifests; install into a project; best‑effort register to Gateway if configured.
* **Gateway:** secure entry; route traffic; list agents/servers; apply auth & policy.

> Keep this as your canonical mental model. It’s 100% compatible with MatrixHub ingestion/installation semantics and MCP Gateway’s `/a2a` + `/servers` APIs.

-----

## Prerequisites

Before proceeding, ensure your development environment meets the following requirements:

  * **Python 3.11** or newer.
  * **Core Development Tools:** `git` and `make`.
  * **(Optional) Containerization:** **Docker** and **Docker Compose** for building and running the service in an isolated environment.
  * **IBM watsonx.ai Credentials:** You will need an active IBM Cloud account. If you don't have one, you can [register for free](https://cloud.ibm.com/registration).
      * `WATSONX_API_KEY`: Your service API key.
      * `WATSONX_URL`: The regional endpoint for your watsonx.ai service (e.g., `https://us-south.ml.cloud.ibm.com`).
      * `WATSONX_PROJECT_ID`: The unique identifier for your watsonx.ai project.
  * **(Optional) Deployment Infrastructure:** A running instance of **MatrixHub** and **MCP Gateway** for the final deployment and registration steps.

> ### **Conceptual Detour: MatrixHub & MCP Gateway**
>
>   * **MatrixHub:** Functions as a **service catalog** or "app store" for distributed agents and tools. It ingests standardized manifests that describe what an agent does and how to communicate with it. Its primary role is discovery and metadata management.
>   * **MCP Gateway:** Acts as a secure, unified **API Gateway**. It provides a single point of entry for clients, routing requests to the appropriate backend agents registered within it. It handles concerns like authentication, routing, and protocol mediation, decoupling clients from the physical location and implementation details of the agents.

-----

## 1\. Project Setup and Installation

First, clone the official repository and set up a virtual environment to manage dependencies.

```bash
# Clone the project repository
git clone https://github.com/ruslanmv/universal-a2a-agent.git
cd universal-a2a-agent

# Create and activate a Python virtual environment
python -m venv .venv
source .venv/bin/activate  # On Windows, use: .venv\Scripts\activate

# Install the core application and its dependencies
pip install -e .

# Install the optional framework adapters used in this tutorial
# The [extras] syntax installs optional dependency groups defined in pyproject.toml
pip install -e .[langgraph]
pip install -e .[langchain]
pip install -e .[crewai]
```

> **Tip:** To install all available adapters at once, you can run `pip install -e .[all]`.

-----

## 2\. Acquiring IBM watsonx.ai Credentials

To configure the watsonx.ai provider, you need to obtain your API key and project details from the IBM Cloud platform.

1.  **Sign in to [IBM Cloud](https://cloud.ibm.com/).**
2.  Navigate to your **watsonx.ai** instance. If you don't have one, create a new project. You can access your projects directly via `https://dataplatform.cloud.ibm.com/projects/`.
3.  Within your project's **Manage** tab, go to the **Access Control** section and create a new **service API key**. Securely copy this key.
4.  Collect the following three values:
      * `WATSONX_API_KEY`: The API key you just generated.
      * `WATSONX_URL`: The regional endpoint shown in your service instance details (e.g., `https://us-south.ml.cloud.ibm.com`).
      * `WATSONX_PROJECT_ID`: The GUID of your project, found in the project's **Manage** -\> **General** settings.

For security and ease of configuration, it is highly recommended to store these credentials in a `.env` file at the root of the project.

**Example `.env` file:**

```env
# Provider and Framework Selection
LLM_PROVIDER=watsonx
AGENT_FRAMEWORK=langgraph

# IBM watsonx.ai Credentials
WATSONX_API_KEY=your_api_key_here
WATSONX_URL=https://us-south.ml.cloud.ibm.com
WATSONX_PROJECT_ID=your_project_guid_here
MODEL_ID=ibm/granite-3-3-8b-instruct # Optional: specify a different model

# Public URL for agent discovery
PUBLIC_URL=http://localhost:8000
```

-----

## 3\. Configuring the Runtime Environment

The agent's behavior is controlled by environment variables. This allows you to switch the active provider or framework without any code changes. Set the following variables in your shell (or confirm they are in your `.env` file).

```bash
# Select the watsonx.ai provider
export LLM_PROVIDER=watsonx

# Set your credentials
export WATSONX_API_KEY=YOUR_KEY
export WATSONX_URL=https://us-south.ml.cloud.ibm.com
export WATSONX_PROJECT_ID=YOUR_PROJECT_ID

# Select the LangGraph framework for orchestration
export AGENT_FRAMEWORK=langgraph  # Other options: crewai, langchain, native
```

-----

## 4\. Local Execution and Verification

With the configuration in place, you can now run the server.

```bash
# The 'make run' command is a convenient shortcut for the uvicorn command
make run

# Alternatively, run uvicorn directly
uvicorn a2a_universal.server:app --host 0.0.0.0 --port 8000
```

Once the server is running, perform a series of smoke tests to verify that all components are operational.

```bash
# 1. Check the interactive API documentation (OpenAPI/Swagger)
open http://localhost:8000/docs

# 2. Check the basic health endpoint (liveness probe)
curl -s http://localhost:8000/healthz

# 3. Check the readiness endpoint (readiness probe), which verifies provider connectivity
curl -s http://localhost:8000/readyz | jq

# 4. Check the agent card for discovery metadata
curl -s http://localhost:8000/.well-known/agent-card.json | jq
```

-----

## 5\. Validating Agent Functionality via Multiple Protocols

Confirm that the agent responds correctly across its different protocol endpoints.

### A) Raw A2A Protocol

```bash
curl -s http://localhost:8000/a2a -H 'Content-Type: application/json' -d '{
  "method":"message/send",
  "params":{"message":{
    "role":"user","messageId":"m1",
    "parts":[{"type":"text","text":"ping from A2A"}]
  }}
}' | jq
```

### B) JSON-RPC 2.0 Protocol

```bash
curl -s http://localhost:8000/rpc -H 'Content-Type: application/json' -d '{
  "jsonrpc":"2.0","id":"1","method":"message/send",
  "params":{"message":{
    "role":"user","messageId":"cli",
    "parts":[{"type":"text","text":"hello via jsonrpc"}]
  }}
}' | jq
```

### C) OpenAI-Compatible Protocol

```bash
curl -s http://localhost:8000/openai/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{
    "model":"universal-a2a-hello",
    "messages":[{"role":"user","content":"hello from openai route"}]
  }' | jq -r '.choices[0].message.content'
```

-----

## 6\. Integration with Orchestration Frameworks

A key advantage of this architecture is its ability to serve as a standardized tool for higher-level frameworks.

> 🚨 **Critical Note on Orchestrator LLMs**
>
> Frameworks like **LangChain** and **CrewAI** often use an LLM for their internal **planning and routing logic** (the "orchestrator"). This is separate from the LLM used for **executing tasks** (our watsonx.ai provider). If the framework's default orchestrator is `ChatOpenAI`, you **must** set an `OPENAI_API_KEY` for the framework to function, even though all substantive work will be routed through our A2A agent to watsonx.ai.

### 6.1. LangChain `Tool` Integration

Here, we wrap our A2A endpoint as a `Tool` that a LangChain agent can decide to use.

```python
# File: examples/quickstart_langchain_watsonx.py
import httpx
from langchain.agents import initialize_agent, AgentType
from langchain_core.tools import Tool
from langchain_openai import ChatOpenAI

BASE = "http://localhost:8000"

# Define the function that calls our A2A endpoint
def a2a_call(prompt: str) -> str:
    try:
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
        # Extract the text response from the A2A message structure
        for p in (data.get("message") or {}).get("parts", []):
            if p.get("type") == "text":
                return p.get("text", "")
        return "[No text part in A2A response]"
    except httpx.HTTPError as e:
        return f"[A2A HTTP Error: {e}]"
    except Exception as e:
        return f"[A2A call failed: {e}]"


# Create the LangChain Tool
tool = Tool(name="a2a_hello", description="Send a prompt to the Universal A2A agent.", func=a2a_call)
llm = ChatOpenAI(model="gpt-4o-mini", temperature=0) # This is the orchestrator LLM
agent = initialize_agent([tool], llm, agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION)

if __name__ == "__main__":
    response = agent.run("Use the a2a_hello tool to say hello to LangChain.")
    print(response)
```

### 6.2. LangGraph `Node` Integration

In LangGraph, our A2A agent can act as a node in a stateful graph.

```python
# File: examples/quickstart_langgraph_watsonx.py
import asyncio, httpx
from langgraph.graph import StateGraph, END, MessagesState
from langchain_core.messages import HumanMessage, AIMessage

BASE = "http://localhost:8000"

async def a2a_send(text: str) -> str:
    try:
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
        return "[No text part in A2A response]"
    except httpx.HTTPError as e:
        return f"[A2A HTTP Error: {e}]"
    except Exception as e:
        return f"[A2A call failed: {e}]"

# This function defines the logic for our graph node
async def a2a_node(state: dict) -> dict:
    last_message = state["messages"][-1]
    reply = await a2a_send(getattr(last_message, "content", ""))
    return {"messages": [AIMessage(content=reply)]}

# Build the graph
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

### 6.3. CrewAI Multi-Agent System

This example demonstrates a multi-agent workflow where agents use our A2A service (backed by watsonx.ai) as their primary tool.

```python
# File: examples/crewai_watsonx_duo.py
import os
import httpx
from crewai import Agent, Task, Crew

BASE = os.getenv("A2A_BASE", "http://localhost:8000")

# The A2A tool is shared by all agents in the crew
def a2a_call(prompt: str) -> str:
    try:
        payload = {
            "method": "message/send",
            "params": {"message": {
                "role": "user", "messageId": "crewai-duo",
                "parts": [{"type": "text", "text": prompt}],
            }},
        }
        r = httpx.post(f"{BASE}/a2a", json=payload, timeout=30.0)
        r.raise_for_status()
        data = r.json()
        for p in (data.get("message") or {}).get("parts", []):
            if p.get("type") == "text":
                return p.get("text", "")
        return "[No text part in A2A response]"
    except Exception as e:
        return f"[A2A call failed: {e}]"


if __name__ == "__main__":
    topic = "Edge AI for autonomous drones in search & rescue"

    # Define the Researcher Agent
    researcher = Agent(
        role="Researcher",
        goal="Gather concise, accurate notes and outline the topic.",
        backstory="Methodical analyst who drafts clean bullet-point notes.",
        tools=[a2a_call],
        allow_delegation=False,
        verbose=True,
    )

    # Define the Writer Agent
    writer = Agent(
        role="Writer",
        goal="Turn notes into a tidy LaTeX article (1–2 pages).",
        backstory="Technical writer who produces compilable LaTeX.",
        tools=[a2a_call],
        allow_delegation=False,
        verbose=True,
    )

    # Define the Research Task
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

    # Define the Writing Task, which depends on the research task
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

    # Assemble and run the crew
    crew = Crew(agents=[researcher, writer], tasks=[t_research, t_write])
    result = crew.kickoff()
    print("\n=== FINAL LATEX ===\n")
    print(result)
```

-----

## 7\. Containerization with Docker

Containerizing the application with Docker ensures consistency and portability across different environments.

**Build and Run the Docker Image:**

```bash
# Build the image, tagging it with a version
docker build -t your-repo/universal-a2a-agent:1.2.0 .

# Run the container, passing credentials as environment variables
docker run --rm -p 8000:8000 \
  -e PUBLIC_URL=http://localhost:8000 \
  -e LLM_PROVIDER=watsonx \
  -e WATSONX_API_KEY=$WATSONX_API_KEY \
  -e WATSONX_URL=$WATSONX_URL \
  -e WATSONX_PROJECT_ID=$WATSONX_PROJECT_ID \
  your-repo/universal-a2a-agent:1.2.0
```

> Note: The `PUBLIC_URL` variable is used by the agent service itself. For deploying the MatrixHub service, a different variable, `PUBLIC_BASE_URL`, is often required.

-----

## 8\. Production Deployment: Publishing to MatrixHub

In a production environment, agents must be discoverable. This is achieved by publishing a manifest to **MatrixHub**.

### 8.1. Authoring the Agent Manifest

Create a manifest file (`agent.manifest.yaml`) that describes your agent. This file contains all the metadata needed for discovery and communication.

```yaml
schema_version: 1

type: agent
id: universal-a2a-hello
version: 1.2.0
name: Universal A2A — Hello
summary: JSON-RPC echo agent, backed by IBM watsonx.ai
homepage: https://example.com/universal-a2a
license: Apache-2.0
capabilities: ["echo", "summarize"]

# A2A-specific manifest block
manifests:
  a2a:
    version: "1.0"
    # This URL must be the final, publicly accessible endpoint of your agent
    endpoint_url: "https://your-agent.example.com/a2a"
    agent_type: jsonrpc # The protocol type
    auth: { type: none, value: "" } # Can be 'bearer' or 'api_key'
    tags: ["watsonx", "demo"]
    # This block allows MatrixHub to auto-register a virtual server in MCP Gateway during install
    server:
      name: universal-a2a-hello-server
      description: Gateway virtual server exposing this A2A agent
```

### 8.2. Publishing to MatrixHub

MatrixHub ingests a catalog by fetching an `index.json` file that points to one or more manifest files. Host your manifest and index files on a reachable `http(s)://` server (e.g., GitHub Pages, S3). `file://` sources are not supported.

**Example `index.json`:**

```json
{
  "manifests": [
    "https://your-host.com/agent.manifest.yaml"
  ]
}
```

**Ingest the catalog into MatrixHub:**

```bash
# Use a non-TLS port for local dev, or https://localhost for local TLS
export HUB_BASE=${HUB_BASE:-http://localhost:8080}

# Ingest using the current route
curl -s -X POST "$HUB_BASE/catalog/ingest" -H 'Content-Type: application/json' \
  -d '{"index_url":"https://your-host.com/index.json"}' | jq

# For legacy deployments, the route might be /ingest
# curl -s -X POST "$HUB_BASE/ingest" -H 'Content-Type: application/json' \
#   -d '{"index_url":"https://your-host.com/index.json"}' | jq
```

> **Note on Database Schema:** For A2A metadata to persist correctly, the MatrixHub database `entity` table requires `protocols` (jsonb) and `manifests` (jsonb) columns. If these are missing, ingestion will still succeed, but A2A-specific data will not be saved.

### 8.3. Installing the Agent and Registering with the Gateway

The **install** step is what triggers the registration with MCP Gateway.

```bash
# This 'install' command triggers the best-effort Gateway registration
curl -s -X POST "$HUB_BASE/catalog/install" -H 'Content-Type: application/json' \
  -d '{"id":"agent:universal-a2a-hello@1.2.0","target":"/tmp/myapp"}' | jq
```

**Corrected Behavior:**
After ingestion, the agent is discoverable in MatrixHub. During the **install** process, if the `MCP_GATEWAY_URL` environment variable is configured for your MatrixHub instance and your `manifests.a2a.server` block is present, MatrixHub will make a best-effort attempt to register the agent with the Gateway via a `POST /a2a` request and create its virtual server via `POST /servers`.

-----

## 9\. Direct Registration with MCP Gateway

For more direct control, you can register the agent and its virtual server programmatically using the production gateway client or raw API calls.

### Recommended Method: Using the Production Client

The preferred, production-safe method is to use the provided gateway client, which handles endpoint logic, idempotency, and authentication.

```python
# File: scripts/register_a2a.py
import os
from src.services.gateway_client import register_a2a_agent, create_server_with_a2a

TOKEN = os.getenv("MCP_GATEWAY_TOKEN")  # optional; client can mint JWT instead

# 1. Define and register the A2A agent
agent_spec = {
    "name": "universal-a2a-hello",
    "endpoint_url": "https://your-agent.example.com/a2a",
    "agent_type": "jsonrpc",
    "auth_type": "none",
    "auth_value": None,
    "tags": ["watsonx", "demo"],
}
print(register_a2a_agent(agent_spec, idempotent=True, token=TOKEN))

# 2. Define and create the virtual server
server_payload = {
    "name": "universal-a2a-hello-server",
    "description": "Gateway virtual server exposing the A2A agent",
    "associated_a2a_agents": [agent_spec["name"]],
}
print(create_server_with_a2a(server_payload, idempotent=True, token=TOKEN))
```

### Alternative: Raw API Requests

For transparency and understanding the underlying API calls, here is the equivalent using raw `requests`. Note the corrected endpoint paths (`/a2a` and `/servers`).

```python
# File: scripts/register_a2a_raw.py
import os, requests

GATEWAY = os.getenv("MCP_GATEWAY_URL", "http://localhost:4444")
HEADERS = {"Content-Type": "application/json"}

# Handle authentication token correctly
if os.getenv("MCP_GATEWAY_TOKEN"):
    t = os.getenv("MCP_GATEWAY_TOKEN").strip()
    HEADERS["Authorization"] = t if t.lower().startswith(("bearer ", "basic ")) else f"Bearer {t}"

# 1. Register agent at POST /a2a
agent = {"name":"universal-a2a-hello","endpoint_url":"https://your-agent.example.com/a2a","agent_type":"jsonrpc","auth_type":"none","auth_value":None,"tags":["watsonx","demo"]}
print(requests.post(f"{GATEWAY}/a2a", json=agent, headers=HEADERS, timeout=15).text)

# 2. Create server at POST /servers
server = {"name":"universal-a2a-hello-server","description":"Gateway virtual server exposing the A2A agent","associated_a2a_agents":[agent["name"]]}
print(requests.post(f"{GATEWAY}/servers", json=server, headers=HEADERS, timeout=15).text)
```

Run your chosen script with your Gateway's URL configured:

```bash
export MCP_GATEWAY_URL="https://gateway.example.com"
# export MCP_GATEWAY_TOKEN="your_token_if_needed"
python scripts/register_a2a.py
```

-----

## 10\. Consuming the Deployed Agent from a watsonx.ai Notebook

Once your agent is deployed and publicly accessible via an HTTPS endpoint, it can be easily consumed from any client environment, including a Jupyter Notebook running in watsonx.ai.

```python
import os, requests

# The public base URL of your deployed agent
BASE = os.getenv("A2A_BASE", "https://your-agent.example.com")
TOKEN = os.getenv("A2A_TOKEN") # An auth token, if you configured one

headers = {"Content-Type": "application/json"}
if TOKEN:
    headers["Authorization"] = f"Bearer {TOKEN}"

body = {
    "model": "universal-a2a-hello",
    "messages": [{"role": "user", "content": "ping from watsonx.ai notebook"}],
}

# Call the OpenAI-compatible endpoint
response = requests.post(
    f"{BASE}/openai/v1/chat/completions",
    json=body,
    headers=headers,
    timeout=20
)
response.raise_for_status()
content = response.json()["choices"][0]["message"]["content"]
print(content)
```

-----

## 11\. Concluding Analysis: Architectural Merits

This architecture provides a robust foundation for building scalable, enterprise-grade agentic applications.

  * **Protocol-Driven Decoupling:** By relying on standard manifests (Agent Card, A2A) and protocols (JSON-RPC, OpenAI-compatible), the agent is decoupled from any specific client or framework, promoting interoperability.
  * **Extensibility and Vendor Neutrality:** The pluggable provider and framework layers allow the system to adapt to new LLMs and orchestration techniques without requiring a rewrite of the core service.
  * **Production-Grade Operations:** The design includes essential operational features like health checks, structured logging, and containerization. The publishing flow via MatrixHub and MCP Gateway provides a managed, secure, and discoverable deployment pattern suitable for enterprise environments.
  * **Simplified Ingestion:** The catalog-based approach with MatrixHub enables a safe and idempotent process for discovering and registering new agents into the ecosystem.