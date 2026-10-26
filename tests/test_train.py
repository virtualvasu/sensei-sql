from pathlib import Path

from conftest import requires_spider

from sensei_sql.paths import CONFIGS
from sensei_sql.train.data import drop_too_long, split
from sensei_sql.train.qlora import load_config


def pairs(n_human, n_teacher):
    return [{"db_id": "d", "question": f"h{i}", "sql": "SELECT 1", "source": "human"} for i in range(n_human)] + [
        {"db_id": "d", "question": f"t{i}", "sql": "SELECT 1", "source": "teacher-synth"} for i in range(n_teacher)
    ]


def test_variant_configs_merge_base():
    for name, data in [("human", ["human"]), ("teacher", ["teacher"]), ("combined", ["human", "teacher"])]:
        cfg = load_config(CONFIGS / "train" / f"{name}.yaml")
        assert cfg["data"] == data
        assert cfg["lora"]["r"] == 16 and cfg["lora"]["lora_alpha"] == 32 and cfg["lora"]["lora_dropout"] == 0.05
        assert cfg["quantization"]["bnb_4bit_quant_type"] == "nf4"
        assert "defaults" not in cfg


def test_split_holds_out_val_and_is_seeded():
    cfg = {"seed": 0, "val_fraction": 0.02, "teacher_fraction": 1.0}
    train, val = split(pairs(500, 500), cfg)
    assert len(val) == 20 and len(train) == 980
    assert not {p["question"] for p in val} & {p["question"] for p in train}
    assert split(pairs(500, 500), cfg) == (train, val)


def test_teacher_fraction_keeps_humans_and_val():
    full_train, full_val = split(pairs(500, 500), {"seed": 0, "val_fraction": 0.02, "teacher_fraction": 1.0})
    train, val = split(pairs(500, 500), {"seed": 0, "val_fraction": 0.02, "teacher_fraction": 0.25})
    assert val == full_val
    n_teacher_full = sum(p["source"] != "human" for p in full_train)
    assert sum(p["source"] == "human" for p in train) == sum(p["source"] == "human" for p in full_train)
    assert sum(p["source"] != "human" for p in train) == round(n_teacher_full * 0.25)


class CharTokenizer:
    """1 token per character; chat template = concatenated contents."""

    def __call__(self, text):
        return {"input_ids": list(text)}

    def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=True):
        return "".join(m["content"] for m in messages)


def test_drop_too_long():
    ex = lambda n: {"prompt": [{"role": "user", "content": "x" * n}], "completion": [{"role": "assistant", "content": "SELECT 1"}]}
    kept, dropped = drop_too_long([ex(10), ex(100), ex(1000)], CharTokenizer(), max_length=200)
    assert len(kept) == 2 and dropped == 1


@requires_spider
def test_human_pairs():
    from sensei_sql.train.data import human_pairs, to_example

    hp = human_pairs()
    assert len(hp) == 8659
    ex = to_example(hp[0])
    assert ex["prompt"][-1]["content"].startswith("Database schema:\nCREATE TABLE")
    assert ex["completion"] == [{"role": "assistant", "content": hp[0]["sql"]}]
    assert "  " not in hp[0]["sql"]
