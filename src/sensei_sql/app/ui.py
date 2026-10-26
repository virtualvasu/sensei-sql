"""Local Gradio interface for the SQL assistant."""

from __future__ import annotations

import pandas as pd

from sensei_sql.app.service import Answer, SQLAssistant, now, timing_text
from sensei_sql.core.executor import ExecResult

HISTORY_COLUMNS = ["time", "database", "question", "sql", "status", "rows"]

STATUS_TEXT = {
    "ok": "✅ {n} row(s){more}",
    "empty": "ℹ️ The query ran but returned no rows.",
    "rejected": "⛔ Only read-only SELECT queries are allowed. {error}",
    "timeout": "⏱️ The query took too long and was stopped. {error}",
    "error": "❌ The query failed: {error}",
    "no_sql": "❌ The model did not produce a SQL query. Try rephrasing the question.",
}


def status_markdown(status: str, result: ExecResult, attempts: list[dict] | None = None) -> str:
    more = f" (showing the first {len(result.rows)})" if result.truncated else ""
    text = STATUS_TEXT[status].format(n=len(result.rows), more=more, error=result.error or "")
    if attempts:
        text += f"\n\n🔁 First attempt failed (`{attempts[0]['error']}`); the model corrected its query."
    return text


def result_frame(result: ExecResult) -> pd.DataFrame:
    if not result.ok or not result.columns:
        return pd.DataFrame()
    cols = [c if c not in result.columns[:i] else f"{c}_{i}" for i, c in enumerate(result.columns)]
    return pd.DataFrame(result.rows, columns=cols)


def build_ui(assistant: SQLAssistant, title: str = "sensei-sql"):
    import gradio as gr

    names = sorted(assistant.databases)
    default = "concert_singer" if "concert_singer" in names else (names[0] if names else None)

    def history_row(db, question, sql, status, n):
        return [now(), db, question, sql, status, n]

    def on_select(db):
        return assistant.schema(db) if db else ""

    def on_upload(file):
        if file is None:
            return gr.update(), ""
        try:
            name = assistant.add_upload(file)
        except ValueError as e:
            return gr.update(), f"❌ Upload rejected: {e}"
        return gr.update(choices=sorted(assistant.databases), value=name), f"✅ Added database `{name}`."

    def on_ask(db, question, history):
        if not db:
            return gr.update(), pd.DataFrame(), "Pick a database first.", "", history, _hist_df(history)
        if not question.strip():
            return gr.update(), pd.DataFrame(), "Type a question first.", "", history, _hist_df(history)
        ans: Answer = assistant.ask(db, question)
        history = history + [history_row(db, question, ans.sql, ans.status, len(ans.result.rows))]
        return (
            ans.sql,
            result_frame(ans.result),
            status_markdown(ans.status, ans.result, ans.attempts),
            timing_text(ans.gen_ms, ans.result.elapsed_ms),
            history,
            _hist_df(history),
        )

    def on_run(db, sql, history):
        if not db or not sql.strip():
            return pd.DataFrame(), "Nothing to run.", "", history, _hist_df(history)
        res = assistant.run(db, sql)
        status = Answer("", db, sql, res).status
        history = history + [history_row(db, "(edited SQL)", sql, status, len(res.rows))]
        return result_frame(res), status_markdown(status, res), timing_text(None, res.elapsed_ms), history, _hist_df(history)

    def _hist_df(history):
        return pd.DataFrame(list(reversed(history)), columns=HISTORY_COLUMNS)

    with gr.Blocks(title=title, analytics_enabled=False) as demo:
        gr.Markdown(f"# {title}\nAsk questions about a SQLite database in plain English. Runs fully offline.")
        history = gr.State([])
        with gr.Row():
            with gr.Column(scale=1):
                db = gr.Dropdown(names, value=default, label="Database", filterable=True)
                upload = gr.File(label="…or upload a SQLite file", file_types=[".sqlite", ".db", ".sqlite3"], type="filepath")
                upload_msg = gr.Markdown()
                with gr.Accordion("Schema", open=False):
                    schema = gr.Code(value=assistant.schema(default) if default else "", language="sql", interactive=False)
            with gr.Column(scale=2):
                question = gr.Textbox(label="Question", placeholder="e.g. How many singers are older than 30?", lines=2)
                ask = gr.Button("Ask", variant="primary")
                sql = gr.Code(label="Generated SQL (editable)", language="sql", interactive=True)
                run = gr.Button("Run edited SQL")
                status = gr.Markdown()
                timing = gr.Markdown()
                table = gr.Dataframe(label="Result", interactive=False, wrap=True)
        with gr.Accordion("Query history (this session)", open=False):
            hist_table = gr.Dataframe(pd.DataFrame(columns=HISTORY_COLUMNS), interactive=False, wrap=True)

        db.change(on_select, db, schema)
        upload.upload(on_upload, upload, [db, upload_msg])
        ask.click(on_ask, [db, question, history], [sql, table, status, timing, history, hist_table])
        question.submit(on_ask, [db, question, history], [sql, table, status, timing, history, hist_table])
        run.click(on_run, [db, sql, history], [table, status, timing, history, hist_table])
    return demo
