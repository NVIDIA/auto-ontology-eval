from ontology_sql_eval.bird_interact.schema import compact_bird_schema


def test_compacts_standard_create_table_schema():
    schema = """\
CREATE TABLE "orders" (
id bigint NOT NULL,
customer_id bigint,
PRIMARY KEY (id),
FOREIGN KEY (customer_id) REFERENCES customers(id)
);

First 3 rows:
id  customer_id
1   7
"""

    assert compact_bird_schema(schema) == "orders: id, customer_id"


def test_compacts_benchmark_schema_with_quoted_sql_keywords():
    schema = """\
"CREATE" TABLE "traders" (
"TR_KEY" text NOT NULL,
"typeFlag" text NULL,
trader_fin_data jsonb NULL,
    "PRIMARY" KEY ("TR_KEY"),
    "UNIQUE" ("typeFlag")
);

"CREATE" TABLE "trades" (
"trade id" bigint NOT NULL,
trader_key text,
    "FOREIGN" KEY (trader_key) REFERENCES traders("TR_KEY"),
    "CHECK" (trader_key <> '')
);
"""

    assert compact_bird_schema(schema) == (
        "traders: TR_KEY, typeFlag, trader_fin_data\n"
        "trades: trade id, trader_key"
    )


def test_compacts_schema_qualified_table_name():
    schema = """\
CREATE TABLE "public"."events" (
event_id bigint
);
"""

    assert compact_bird_schema(schema) == "events: event_id"

