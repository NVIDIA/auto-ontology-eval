# Auto Ontology Agent Server (the ReAct arm)

[← Back to main README](../../README.md)

This NeMo Gym Agent Server runs Auto Ontology's ReAct text-to-SQL agent
(`auto_ontology.retrieval.text_to_sql.react`) as the agent of a Gym
environment. The policy model is the agent's model, so its rollouts can be
compared with the schema-only arm and used for training.

The treatment arm (`auto_ontology_sql`) is different: it runs Auto Ontology
inside a Resources Server's `verify()` and discards the policy's output.

## How an episode runs

```
Environment Server (single_agent_turn)
  seed     ─▶ Agent Server: start a sandbox from the agent image, start its catalog Postgres
  activate ─▶ Agent Server: run one ReAct episode in the sandbox
                 sandbox ─▶ policy Model Server (each model call, on the rollout's route)
                 sandbox ─▶ catalog + source database (retrieve, run_sql, submit)
              ◀─ the submitted query, in a ```sql block
  close    ─▶ Agent Server: stop the sandbox
  verify   ─▶ schema_only_sql Resources Server: exec_match against the gold query
```

- **One sandbox per episode.** The image holds Auto Ontology, this repository,
  the BIRD dev databases and a catalog that is already ingested and compiled.
  Episodes do not share state.
- **The agent's process imports only NeMo Gym.** Auto Ontology runs in the
  sandbox (`ontology_sql_eval/gym/react_runner.py`).
- **The episode is Auto Ontology's own.** The runner calls `run_episode` as
  Auto Ontology's `ReactSQLAgent` does, with SQL answers only, the question as
  asked, and the evidence apart. The agent ignores the activation's prompt.
- **The grader is the schema-only arm's.** `schema_only_sql` grades the query
  with the same `exec_match` rules.

## Build the image

The image is built during the benchmark setup, once for each Auto Ontology
revision:

```bash
scripts/build_agent_image.sh .env   # tag: auto-ontology-eval/agent:bird
```

The script seeds BIRD dev if it is not there. Then it runs this repository's
pipeline (`main.py --database-name bird --skip-eval`: migration, ingest and
semantic compile) inside the image. The model keys in `.env` reach that step
only as a BuildKit secret, so no image layer holds them.

The build calls the embedding and compile models for all 11 databases. If a
database has no vectors after the build, the build fails. Without that check,
`main.py` would log a missing key and continue.

Retrieval embeds each question during the episode. Thus the episode needs the
key of the same embedding endpoint. The image records the endpoint and model
names in `.env`. The key comes from the agent's config.

## Run

1. Build the tasks:

   ```bash
   uv run python scripts/build_gym_tasks.py --dataset bird --arm react --limit 200
   ```

   The rows carry the question, the evidence and the database in their
   `agent_context`, and no schema dump.

2. Run the arm. Give the embedding key as `auto_ontology_api_key` in
   `env.yaml` or as an override:

   ```bash
   gym eval run --search-dir . \
     --config resources_servers/schema_only_sql/configs/schema_only_sql.yaml \
     --config responses_api_agents/auto_ontology_agent/configs/auto_ontology_agent.yaml \
     --model-type inference_provider --model <model> --model-url <url> \
     --split validation --output runs/react/bird.jsonl \
     ++auto_ontology_api_key=<key> \
     ++default_host=$(docker network inspect bridge -f '{{(index .IPAM.Config 0).Gateway}}')
   ```

   The sandbox sends model calls to the Model Server's address. The config
   sets `use_absolute_ip: true`. If the host name resolves to a loopback
   address, set `default_host` to an address that the sandbox can reach, such
   as Docker's bridge gateway.

## Settings

| Setting | Default | Meaning |
|---|---|---|
| `model_server` | `policy_model` | The Model Server the episode runs on |
| `model` | `openai/policy` | The name Auto Ontology sends. The `openai/` prefix selects its OpenAI-compatible client. The Model Server uses its own model name. |
| `sandbox_provider`, `sandbox_config` | `auto_ontology_sandbox`, the image | The sandbox of each episode |
| `environment` | `DEFAULT_MODELS_API_KEY` | Auto Ontology's settings in the sandbox, such as `TEXT_TO_SQL_REACT_MAX_TOOL_CALLS`. These settings cannot change the model. |
| `setup_timeout_seconds`, `episode_timeout_seconds`, `close_timeout_seconds` | 300, 1800, 60 | The time limits for the catalog start, the episode and the close |
| `max_concurrent_episodes` | 8 | The number of episodes that run at the same time in this server. Each episode has its own sandbox and Postgres. |

## Known limits

- Usage stays in the Model Server's capture (`+observability_enabled=true`).
  The agent's reply has no `usage` field, and the close reports the
  `model_call_usage_unavailable` gap.
- An activation that fails is not retried in the same sandbox. The
  Environment Server's retry starts a new episode.
- Only BIRD dev is in the image. Other datasets need their databases and a
  catalog in an image of their own.
