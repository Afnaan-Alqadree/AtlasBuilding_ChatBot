# AtlasBuilding Chatbot — Architecture & Flow

---

## 1) System Context (overview)
```mermaid
graph LR
    U["Users: Facility mgrs • Students • Researchers"]
    UI["Streamlit Chat UI\n(src/ui.py)"]
    C["Chatbot Coordinator\n(src/chatbot.py)"]
    R["NL Router (4 routes)\n(src/router.py)"]

    subgraph Data Tier
      DB["DuckDB (read‑only)\n(src/db.py)\n• Views over CSVs\n• Safety: SELECT/WITH+LIMIT"]
      CSV1["Atlas_occupancy_sensors_offices.csv"]
      CSV2["Atlas_occupancy_sensors_meeting_rooms.csv"]
      CSV3["sensor_space_data_occupancy.csv"]
      DB --> CSV1
      DB --> CSV2
      DB --> CSV3
    end

    subgraph Intelligence Tier
      direction TB
      LLM["LLM Client wrapper\n(src/llm_client.py)\n• OpenAI or Ollama (local)"]
      AG["Agent (heuristic / LC)\n(src/agent.py | src/agent_lc.py)"]
      TOOLS["Planner & Tools\n(src/planner.py | agent_tools.py)\n• e.g., Coffee/Toilet placement"]
      subgraph RAG
        direction TB
        CH["Chroma Vector Store\n(src/rag.py)"]
        GP["Grounding pack\n( atlas_grounding_pack.json )"]
        DP["Data profile builder\n(src/data_profile.py)"]
        PR["Prompt builder\n(src/prompts.py)"]
        DP --> GP
        CH --> PR
      end
      AG --> LLM
      AG -. may retrieve .-> CH
      AG -. may call .-> TOOLS
      AG -. may emit .->|Safe SQL| DB
    end

    U --> UI --> C --> R

    %% Routing branches
    R -->|1) tool| TOOLS --> DB
    R -->|2) sql (template)| DB
    R -->|3) agent| AG
    R -->|4) LLM routing| LLM

    %% Evidence back to UI
    DB -. results + SQL .-> UI
    AG -. answer + citations .-> UI
    TOOLS -. analytics .-> UI

    %% Future (optional)
    subgraph Future (dashed)
      style Future stroke-dasharray: 5 5
      PM["Predictive models\n(occupancy/energy)\nscikit‑learn / LSTM"]
    end
    PM -. planned .-> AG
```

**Notes**
- The router picks the lightest correct path: fast tools → template SQL → agent → LLM routing.
- All SQL is read‑only and safety‑checked. UI shows the SQL and a sample of rows as evidence.
- Agent can optionally use RAG for “why/how” explanations and call domain tools (e.g., placement planners).

---

## 2) Component Breakdown
```mermaid
graph TB
  subgraph UI Layer
    UI["Streamlit UI\nui.py"]
  end

  subgraph Orchestration
    CB["Chatbot\nchatbot.py"]
    RT["Router (intents)\nrouter.py"]
  end

  subgraph Data Access & Analytics
    DB["DuckDB client\ndb.py"]
    VW["Materialized views\n(SELECT/WITH only)"]
  end

  subgraph Intelligence
    LC["LLM Client\nllm_client.py"]
    AG["Agent / Agent LC\nagent.py / agent_lc.py"]
    PL["Planners & Tools\nplanner.py / agent_tools.py"]
  end

  subgraph Retrieval
    RG["RAG builder/loader\nrag.py"]
    CH["Chroma store\n(.chroma/)"]
    PR["Prompt builder\nprompts.py"]
    GP["Grounding JSON\natlas_grounding_pack.json"]
    DP["Data profile\ndata_profile.py"]
  end

  UI --> CB --> RT
  RT -->|tool| PL --> DB
  RT -->|sql| DB
  RT -->|agent| AG --> LC
  AG -. may use .-> RG --> CH
  DP --> GP --> PR
  PR -. system prompt .-> LC
  DB --> VW
  DB -. results + SQL .-> UI
```

---

## 3) Sequence — “Utilization by floor (last 30 days)”
```mermaid
sequenceDiagram
  actor User
  participant UI as Streamlit UI (ui.py)
  participant CB as Chatbot (chatbot.py)
  participant RT as Router (router.py)
  participant DB as DuckDB (db.py)
  participant AG as Agent (agent.py)
  participant LC as LLM Client
  participant RG as RAG (rag.py)

  User->>UI: Ask question
  UI->>CB: Pass query
  CB->>RT: Route intent
  RT-->>CB: Decide: template SQL
  CB->>DB: Execute safe SQL (SELECT/WITH + LIMIT)
  DB-->>CB: Rows + SQL string
  CB-->>UI: Table + “Route: SQL” + evidence
  note over UI: User sees result + SQL for transparency

  alt If question is “why/how/describe”
    RT-->>CB: Decide: agent
    CB->>AG: Plan & reason
    AG->>LC: Call LLM (with system prompt)
    opt Needs context
      AG->>RG: Retrieve chunks from Chroma
      RG-->>AG: Context passages
    end
    AG-->>CB: Answer + citations/tools/SQL
    CB-->>UI: Explanatory answer + evidence
  end
```

---

## 4) Data & Safety Summary (for documentation)
- **Datasets**: `Atlas_occupancy_sensors_offices.csv`, `Atlas_occupancy_sensors_meeting_rooms.csv`, `sensor_space_data_occupancy.csv`.
- **Database**: DuckDB with read‑only helpers; only `SELECT/WITH` allowed + single statement + `LIMIT` enforced.
- **Routing**: tool → SQL → agent → LLM; prefer simpler/cheaper routes.
- **Explainability**: UI shows SQL and sample rows; agent can include citations from RAG.
- **Extensibility**: plug new tools/planners (e.g., free rooms, placement suggestions).
- **Models**: OpenAI or local (Ollama) via `llm_client.py`.
- **Grounding**: `data_profile.py` builds `atlas_grounding_pack.json`; `prompts.py` composes the system prompt.
- **Future work**: predictive models to forecast occupancy/energy.

