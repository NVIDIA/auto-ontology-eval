# Evaluation flow, now and next

## Now: paired arms on NeMo Gym

```mermaid
flowchart LR
  seed[("Seed a benchmark database: BIRD, BEAVER, FDABench-Lite, Wide World Importers")]
  ingest["Ingest the schema into Auto-Ontology's catalog"]
  compile["Compile the ontology"]
  control["Control arm: the model gets the question and a schema dump"]
  treat["Treatment arm: the same model, through Auto-Ontology's agent"]
  verify["Verifier: run each arm's SQL and the gold SQL, compare results, no LLM"]
  report["Report: accuracy per arm, the paired difference, questions fixed and broken"]
  seed --> ingest --> compile --> treat --> verify
  seed --> control --> verify
  verify --> report
```

Each question runs twice with the same model: the control arm
(`schema_only_sql`) gets the question and a schema dump in one prompt, and the
treatment arm (`auto_ontology_sql`) runs Auto-Ontology's agent in-process at
the revision pinned in [`pyproject.toml`](../pyproject.toml). One verifier
(`ontology_sql_eval/gym/exec_match.py`) runs each arm's SQL and the gold SQL and
compares the results by each benchmark's official rule, with no LLM. Details
are in [resources_servers/README.md](../resources_servers/README.md).

In the treatment arm, Gym still calls its policy model and discards the answer;
Auto-Ontology's agent writes the SQL, and its own model calls are routed
through Gym model servers so their tokens are counted
(`ontology_sql_eval/gym/model_routing.py`).

## Next: ReAct tools on NeMo Gym

**Planned, not built yet.** Now, Gym's policy model answers only in the
control arm; in the treatment arm its answer is discarded and Auto-Ontology's
agent writes the SQL. With the ReAct agent, the plan is for the policy model to
call Auto-Ontology's tools directly, with the same verifier scoring each
submitted query as the reward.

```mermaid
flowchart LR
  task["Gym task: question, database, gold SQL"]
  policy{"Policy model: choose the next tool"}
  subgraph env["Auto-Ontology tools as the Gym environment"]
    direction TB
    retrieve["retrieve: search the ontology"]
    runsql["run_sql: check names, run read-only"]
    submit["submit: check and run the final query"]
  end
  verify["Verifier: run submitted and gold SQL, compare results, no LLM"]
  score["Score: accuracy per question, the reward for training"]
  traj[("Trajectory: every message and tool call")]
  task --> policy
  policy -- "tool call" --> env
  env -- "result" --> policy
  policy -- "submitted query" --> verify --> score
  policy -.-> traj
  classDef model fill:#eaf5d6,stroke:#76b900,color:#111
  classDef det fill:#f2f2f2,stroke:#777,color:#111
  classDef react fill:#efe6f7,stroke:#7b3fa0,color:#111
  class policy react
  class retrieve model
  class runsql,submit,verify det
```

Green: model call. Gray: deterministic (SQL or rules). Purple: model-chosen tool (ReAct). Dashed: off by default or optional.
