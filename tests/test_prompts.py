from sensei_sql.core.prompts import (
    PATTERNS,
    answer_messages,
    extract_sql,
    generate_pairs_messages,
    parse_generated_pairs,
    student_messages,
)


def test_student_prompt_is_canonical_answer_prompt():
    msgs = student_messages("CREATE TABLE t (a INT);", "How many rows?")
    assert [m["role"] for m in msgs] == ["system", "user"]
    assert msgs == answer_messages("CREATE TABLE t (a INT);", "How many rows?")
    assert "CREATE TABLE t (a INT);" in msgs[1]["content"]
    assert msgs[1]["content"].endswith("Question: How many rows?")


def test_few_shot_turns_precede_question():
    msgs = answer_messages("S", "Q", examples=[("S1", "Q1", "SELECT 1"), ("S2", "Q2", "SELECT 2")])
    assert [m["role"] for m in msgs] == ["system", "user", "assistant", "user", "assistant", "user"]
    assert msgs[2]["content"] == "SELECT 1"
    assert msgs[-1] == student_messages("S", "Q")[-1]


def test_generate_prompt_mentions_pattern():
    for key, desc in PATTERNS.items():
        assert desc in generate_pairs_messages("S", key, 5)[1]["content"]


def test_parse_generated_pairs():
    text = (
        "<think>planning</think>\n"
        '{"question": "How many  singers?", "sql": "SELECT count(*) FROM singer;"}\n'
        "not json\n"
        '{"question": "", "sql": "SELECT 1"}\n'
        '{"question": "Names?", "sql": "```sql\\nSELECT name FROM singer\\n```"},\n'
    )
    assert parse_generated_pairs(text) == [
        {"question": "How many singers?", "sql": "SELECT count(*) FROM singer"},
        {"question": "Names?", "sql": "SELECT name FROM singer"},
    ]


def test_extract_sql():
    cases = {
        "```sql\nSELECT  a,\n  b FROM t;\n```": "SELECT a, b FROM t",
        "SELECT 1; DROP TABLE t": "SELECT 1",
        "The query is: SELECT x FROM t": "SELECT x FROM t",
        "With this schema the answer is:\nSELECT x FROM t": "SELECT x FROM t",
        "<think>SELECT wrong</think>\nWITH c AS (SELECT 1) SELECT * FROM c": "WITH c AS (SELECT 1) SELECT * FROM c",
        "SELECT 'a;  b' FROM t WHERE c = \"x  y\"": "SELECT 'a;  b' FROM t WHERE c = \"x  y\"",
        "SELECT 'it''s' FROM t;": "SELECT 'it''s' FROM t",
        "```sql\nSELECT count(*) FROM singer": "SELECT count(*) FROM singer",
        "I cannot answer that.": "",
    }
    for text, want in cases.items():
        assert extract_sql(text) == want, text
