from ontology_sql_eval.bird_interact.budget import Budget, MeteredConnector


class _Base:
    def execute(self, sql):
        return [(1,)]


def _connector(left: float, phase: int = 1) -> tuple[MeteredConnector, Budget]:
    budget = Budget.start(left)
    return MeteredConnector(
        _Base(), budget, lambda sql, charged: None, phase=lambda: phase
    ), budget


def test_phase1_status_shows_the_balance_and_asks_to_keep_a_phase2_reserve():
    connector, _ = _connector(12.0)

    status = connector.cost_status()

    assert status.startswith("BUDGET: 12 coins left.")
    assert "run_sql costs 1 coin" in status
    assert "at least 6 coins unspent" in status


def test_status_follows_the_balance_as_queries_are_charged():
    connector, _ = _connector(12.0)

    connector.execute("SELECT 1")

    assert connector.cost_status().startswith("BUDGET: 11 coins left.")


def test_phase2_status_has_no_reserve():
    connector, _ = _connector(5.0, phase=2)

    status = connector.cost_status()

    assert status.startswith("BUDGET: 5 coins left for this follow-up.")
    assert "unspent" not in status


def test_status_says_to_submit_when_no_query_is_affordable():
    connector, _ = _connector(0.5)

    assert "Submit your best query now" in connector.cost_status()


def test_queries_stop_being_runnable_when_the_last_coin_is_spent():
    connector, _ = _connector(1.0)

    assert connector.can_run_queries()
    connector.execute("SELECT 1")
    assert not connector.can_run_queries()
