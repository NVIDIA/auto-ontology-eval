# Analytics shipping: synthetic ontology task fixture v1

This is a small, offline acceptance fixture for ontology-aware analytics and future task generation. It is **not** a TaskGen implementation, deployed catalog, or benchmark score. No model keys, model calls, or external services are needed to validate it.

- `source.sql`: SQLite DDL and fabricated order/parcel data. `order_date` and `shipped_date` are ISO calendar dates; the October window is `[2026-10-01, 2026-11-01)`.
- `model.yaml`: native Auto Ontology model document with fixed fixture IDs. It defines **Paid Order Net Revenue** in integer USD cents at order grain. `shipments.order_id → orders.id` is many-to-one, so filter orders with `EXISTS` rather than summing after joining to shipment rows. This reference definition is hand-checked for the fixture, not certified in a deployed service.
- `questions.json`: common inputs to both arms. Give the schema-only arm only the physical DDL, not row inserts or the ontology. Give the ontology arm the same questions and source data, with `model.yaml` as the reference overlay for an isolated catalog. The oracle is never a policy input.
- `oracle.json`: fixed expected outcome, reference SQL and exact rows where appropriate. The four held-out cases cover the metric, fanout and the time axis, a question that needs clarification, and missing cost data that cannot be treated as zero.

Run `uv run pytest tests/test_analytics_shipping_fixture.py`. This checks the native model shape against the physical SQLite tables and executes the reference queries against the fixture. It does not import or compile the catalog into a running service, call MCP, or grant a training reward. The questions here are hand-authored acceptance cases; generate distinct schemas or task families for training rather than reusing them.
