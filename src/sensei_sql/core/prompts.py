"""Prompt templates shared by the teacher, the student and the app, plus SQL extraction.

Everything that answers a question with SQL (student training, student inference, teacher
relabelling, the app) goes through :func:`answer_messages`, so the schema format and wording
are identical everywhere. Messages are plain ``[{"role", "content"}]`` lists; render them with
the model's chat template (``tokenizer.apply_chat_template`` or ``vllm.LLM.chat``). For the
Qwen3.5 teacher pass ``chat_template_kwargs={"enable_thinking": False}`` when calling it.
"""

from __future__ import annotations

import json
import re
from collections.abc import Sequence

Messages = list[dict[str, str]]

ANSWER_SYSTEM = (
    "You are an expert SQLite developer. Given a database schema and a question, write one "
    "SQLite SELECT query that answers the question. Reply with the SQL query only, no explanation."
)


def answer_user_content(schema: str, question: str) -> str:
    return f"Database schema:\n{schema}\n\nQuestion: {question}"


def answer_messages(
    schema: str,
    question: str,
    examples: Sequence[tuple[str, str, str]] = (),
) -> Messages:
    """Schema + question -> SQL. ``examples`` are few-shot ``(schema, question, sql)`` triples,
    given as earlier chat turns (teacher few-shot only; must come from train databases)."""
    messages: Messages = [{"role": "system", "content": ANSWER_SYSTEM}]
    for ex_schema, ex_question, ex_sql in examples:
        messages.append({"role": "user", "content": answer_user_content(ex_schema, ex_question)})
        messages.append({"role": "assistant", "content": ex_sql})
    messages.append({"role": "user", "content": answer_user_content(schema, question)})
    return messages


def student_messages(schema: str, question: str) -> Messages:
    """The canonical student prompt (zero-shot). The training target is the bare SQL string."""
    return answer_messages(schema, question)


def render_prompt(tokenizer, messages: Messages, **template_kwargs) -> str:
    """Apply a Hugging Face tokenizer's chat template, ready for generation."""
    return tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True, **template_kwargs
    )


# --- Teacher: synthesise question/SQL pairs ---------------------------------------------------

# SQL patterns used to steer synthesis toward diverse queries (#4).
PATTERNS: dict[str, str] = {
    "single_table": "single-table queries with WHERE filters (comparisons, LIKE, IN, BETWEEN, NULL checks)",
    "aggregation": "aggregation with COUNT/SUM/AVG/MIN/MAX, GROUP BY and HAVING",
    "join": "queries that JOIN two or more tables through their foreign keys",
    "nested": "nested subqueries (IN / NOT IN / EXISTS / scalar subqueries in WHERE or SELECT)",
    "order_limit": "ORDER BY ... LIMIT queries (top-k, superlatives such as highest/lowest/most)",
    "set_ops": "set operations combining two SELECTs with UNION, INTERSECT or EXCEPT",
}

GENERATE_SYSTEM = (
    "You are an expert SQLite developer creating a text-to-SQL training dataset. You write "
    "realistic natural-language questions a user might ask about a database, each paired with "
    "a correct SQLite SELECT query that answers it."
)


def generate_pairs_messages(schema: str, pattern: str, n: int) -> Messages:
    """Ask the teacher for ``n`` question/SQL pairs of one :data:`PATTERNS` key over ``schema``."""
    user = (
        f"Database schema:\n{schema}\n\n"
        f"Write {n} different question/SQL pairs for this database. Every query must use: "
        f"{PATTERNS[pattern]}.\n"
        "Rules:\n"
        "- Use only tables and columns that exist in the schema above.\n"
        "- Questions must be answerable from the data and must not mention SQL or column names verbatim when a natural phrasing exists.\n"
        "- Each SQL is a single SQLite SELECT statement.\n"
        'Output one JSON object per line, exactly: {"question": "...", "sql": "..."}\n'
        "Output nothing else."
    )
    return [{"role": "system", "content": GENERATE_SYSTEM}, {"role": "user", "content": user}]


def parse_generated_pairs(text: str) -> list[dict[str, str]]:
    """Parse teacher output from :func:`generate_pairs_messages`; skips malformed lines."""
    pairs = []
    for line in strip_think(text).splitlines():
        line = line.strip().rstrip(",")
        if not line.startswith("{"):
            continue
        try:
            obj = json.loads(line)
        except json.JSONDecodeError:
            continue
        q, sql = obj.get("question"), obj.get("sql")
        if isinstance(q, str) and isinstance(sql, str) and q.strip() and sql.strip():
            pairs.append({"question": " ".join(q.split()), "sql": extract_sql(sql)})
    return pairs


# --- SQL extraction ----------------------------------------------------------------------------

_THINK = re.compile(r"<think>.*?</think>", re.S)
_FENCE = re.compile(r"```[A-Za-z]*\s*\n?(.*?)```", re.S)
# Prefer a statement starting a line, so prose like "a query with a join" isn't taken as WITH.
_SQL_LINE_START = re.compile(
    r"^[ \t]*(SELECT\b|WITH\s+(RECURSIVE\s+)?[\w\"`\[\]]+\s*(\([^)]*\))?\s*AS\s*\()", re.I | re.M
)
_SELECT_ANYWHERE = re.compile(r"\bSELECT\b", re.I)


def strip_think(text: str) -> str:
    text = _THINK.sub("", text)
    # An unterminated <think> (generation cut off) leaves no usable answer before it.
    return text.split("</think>")[-1] if "</think>" in text else text


def extract_sql(text: str) -> str:
    """Pull one SQL statement out of model output.

    Strips ``<think>`` blocks and code fences, skips leading prose up to the first
    SELECT/WITH, keeps only the first statement, and collapses whitespace outside string
    literals. Returns ``""`` if nothing is found.
    """
    text = strip_think(text)
    fence = _FENCE.search(text)
    if fence:
        text = fence.group(1)
    elif "```" in text:  # unterminated fence
        text = text.split("```", 1)[1].split("\n", 1)[-1]
    m = _SQL_LINE_START.search(text) or _SELECT_ANYWHERE.search(text)
    if not m:
        return ""
    return _first_statement_normalised(text[m.start():])


def _first_statement_normalised(sql: str) -> str:
    """Return the text up to the first ``;`` outside quotes, with whitespace collapsed outside quotes."""
    out: list[str] = []
    quote: str | None = None
    pending_space = False
    for ch in sql:
        if quote:
            out.append(ch)
            if ch == quote:
                quote = None
            continue
        if ch == ";":
            break
        if ch.isspace():
            pending_space = bool(out)
            continue
        if pending_space:
            out.append(" ")
            pending_space = False
        out.append(ch)
        if ch in "'\"`":
            quote = ch
        elif ch == "[":
            quote = "]"
    return "".join(out).strip()
