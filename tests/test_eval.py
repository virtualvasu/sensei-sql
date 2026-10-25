import json

from conftest import requires_spider

from sensei_sql.eval.aggregate import collect, to_markdown
from sensei_sql.eval.runner import percentile


def test_percentile():
    vals = [float(v) for v in range(1, 101)]
    assert percentile(vals, 50) == 50.5
    assert 95 <= percentile(vals, 95) <= 96
    assert percentile([7.0], 95) == 7.0


def test_aggregate(tmp_path):
    for run, ex in [("a", 0.5), ("b", 0.75)]:
        (tmp_path / run).mkdir()
        level = {"count": 4, "correct": int(4 * ex), "exec_accuracy": ex}
        metrics = {
            "run": run,
            "model": "student",
            "adapter": None,
            "few_shot": [],
            "num_questions": 4,
            "accuracy": {"by_hardness": {h: level for h in ("all", "easy", "medium", "hard", "extra")}},
            "efficiency": {"latency_ms_p50": 100.0, "peak_gpu_mem_gib": 6.5},
        }
        (tmp_path / run / "metrics.json").write_text(json.dumps(metrics))
    df = collect(tmp_path)
    assert list(df["run"]) == ["a", "b"] and list(df["ex_all"]) == [0.5, 0.75]
    md = to_markdown(df)
    assert "| a | student | 0 | 4 | 0.500 |" in md and "| – |" in md  # missing metrics shown as –


@requires_spider
def test_scoring_matches_gold_and_rejects_unsafe():
    from sensei_sql.eval.data import load_split
    from sensei_sql.eval.score import score_rows

    rows = [
        {"db_id": e.db_id, "question": e.question, "gold_sql": e.gold_sql, "pred_sql": e.gold_sql}
        for e in load_split("dev", limit=30)
    ]
    rows[0]["pred_sql"] = "DROP TABLE singer"
    rows[1]["pred_sql"] = ""
    rows[2]["pred_sql"] = "SELECT count(*) FROM stadium"
    m = score_rows(rows, "dev", timeout_s=5)
    assert [r["exec_match"] for r in rows[:3]] == [False, False, False]
    assert all(r["exec_match"] for r in rows[3:])
    assert m["by_hardness"]["all"] == {"count": 30, "correct": 27, "exec_accuracy": 0.9}
    assert sum(m["by_hardness"][h]["count"] for h in ("easy", "medium", "hard", "extra")) == 30


@requires_spider
def test_few_shot_never_uses_eval_databases():
    from sensei_sql.eval.data import load_split, select_few_shot
    from sensei_sql.eval.testsuite import hardness

    dev_dbs = {e.db_id for e in load_split("dev")}
    shots = select_few_shot(["easy", "medium", "hard"], 600, 0, dev_dbs)
    assert [hardness(s.db_path, s.gold_sql) for s in shots] == ["easy", "medium", "hard"]
    assert not {s.db_id for s in shots} & dev_dbs
    assert len({s.db_id for s in shots}) == 3
    assert shots == select_few_shot(["easy", "medium", "hard"], 600, 0, dev_dbs)  # deterministic
