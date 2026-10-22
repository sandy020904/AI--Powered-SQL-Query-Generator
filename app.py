"""SQL Query Generator Using LLMs, SQLite & Generative AI - Streamlit UI."""
from __future__ import annotations

from datetime import datetime

import altair as alt
import pandas as pd
import streamlit as st

import database as db
import llm_service as llm
from sql_validator import validate_sql

st.set_page_config(page_title="SQL Query Generator", page_icon="🗄️", layout="wide")

MAX_AUTO_FIX = 2
ACCENT = "#0b7a75"

CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;500&family=IBM+Plex+Sans:wght@400;500;600&display=swap');
:root { --ink:#16233b; --muted:#5b6b82; --line:#d5dde8; --accent:#0b7a75; }
html, body, .stApp, .stMarkdown, button, input, textarea { font-family:'IBM Plex Sans','Segoe UI',sans-serif; }
.block-container { padding-top:2rem; max-width:1200px; }
section[data-testid="stSidebar"] { border-right:1px solid var(--line); }
.app-title { font-size:2.1rem; font-weight:600; letter-spacing:-0.02em; margin:0; color:var(--ink); }
.app-sub { color:var(--muted); margin:.3rem 0 1.4rem; font-size:1.02rem; }
.st-key-sql_editor textarea {
  font-family:'IBM Plex Mono','Consolas',monospace; font-size:.92rem; line-height:1.55;
  background:#13202f; color:#e4ecf6; caret-color:#6ee0d6;
  border-left:4px solid var(--accent); border-radius:6px;
}
[data-testid="stMetric"] { background:#fff; border:1px solid var(--line); padding:.7rem 1rem; border-radius:6px; }
.empty { border:1px dashed var(--line); border-radius:6px; padding:1.4rem; color:var(--muted); background:#fff; }
@media (max-width: 700px) { .app-title { font-size:1.6rem; } .block-container { padding:1rem .8rem; } }
</style>
"""
st.markdown(CSS, unsafe_allow_html=True)


# ------------------------------------------------------------------ state
def _init_state() -> None:
    defaults = {
        "history": [], "question": "", "sql_editor": "", "result": None, "error": None,
        "explanation": None, "notice": None, "active_question": "", "upload_n": 0,
    }
    for key, value in defaults.items():
        st.session_state.setdefault(key, value)


def notify(kind: str, text: str) -> None:
    st.session_state.notice = (kind, text)


def reset_workspace() -> None:
    st.session_state.update(sql_editor="", result=None, error=None, explanation=None, active_question="")


def request(action: str) -> None:
    st.session_state.pending = action


def use_example(text: str) -> None:
    st.session_state.question = text


def reuse(entry: dict) -> None:
    st.session_state.update(
        question=entry["question"] if entry["question"] != "Manual SQL" else "",
        sql_editor=entry["sql"], active_question=entry["question"], result=None, error=None, explanation=None,
    )


def add_uploaded_db() -> None:
    f = st.session_state.get(f"db_upload_{st.session_state.upload_n}")
    if f is None:
        return notify("warning", "Choose a SQLite file first.")
    try:
        name = db.save_uploaded_database(f.name, f.getvalue())
    except db.DatabaseError as exc:
        return notify("error", str(exc))
    st.session_state.db_name = name
    st.session_state.upload_n += 1
    reset_workspace()
    notify("success", f"Added database '{name}'.")


def add_csv_db() -> None:
    f = st.session_state.get(f"csv_upload_{st.session_state.upload_n}")
    if f is None:
        return notify("warning", "Choose a CSV file first.")
    try:
        name = db.create_database_from_csv(f.name, f.getvalue())
    except db.DatabaseError as exc:
        return notify("error", str(exc))
    st.session_state.db_name = name
    st.session_state.upload_n += 1
    reset_workspace()
    notify("success", f"Created database '{name}' from CSV.")


def rebuild_sample() -> None:
    db.ensure_sample_database(overwrite=True)
    st.session_state.db_name = db.SAMPLE_DB_NAME
    reset_workspace()
    notify("success", "Sample database rebuilt.")


# ---------------------------------------------------------------- helpers
def example_questions(schema: list[db.TableInfo], is_sample: bool) -> list[str]:
    if is_sample:
        return [
            "Show the top 5 customers by total spending",
            "What is the monthly revenue for delivered orders in 2025?",
            "Which product categories generate the most revenue?",
            "List the 10 best-selling products by quantity sold",
            "How many orders are there in each status?",
            "Which customers have never placed an order?",
            "What is the average order value per country?",
            "Show products with fewer than 20 items in stock",
        ]
    if not schema:
        return []
    first = schema[0].name
    return [f"Show the first 10 rows of {first}", f"How many rows are in each table?",
            f"What are the distinct values in the first column of {first}?"]


def table_names(schema: list[db.TableInfo]) -> list[str]:
    return [t.name for t in schema]


def log_history(sql: str, ok: bool, rows: int = 0, error: str = "") -> None:
    st.session_state.history.insert(0, {
        "time": datetime.now().strftime("%H:%M:%S"),
        "question": st.session_state.active_question or "Manual SQL",
        "sql": sql, "ok": ok, "rows": rows, "error": error,
    })
    del st.session_state.history[100:]


# --------------------------------------------------------- pending actions
def do_generate(schema: list[db.TableInfo]) -> None:
    question = st.session_state.question.strip()
    if not question:
        return notify("warning", "Type a question first, or pick an example from the sidebar.")
    try:
        with st.spinner("Writing SQL…"):
            sql = llm.generate_sql(question, db.schema_to_prompt(schema, question))
    except llm.LLMError as exc:
        return notify("error", str(exc))
    st.session_state.update(sql_editor=sql, result=None, error=None, explanation=None, active_question=question)
    check = validate_sql(sql, table_names(schema))
    if check.ok:
        notify("success", "SQL generated. Review it below, edit if needed, then run it.")
    else:
        notify("warning", f"The generated SQL can't be run as is: {check.error} Edit it or ask again.")


def do_execute(path, schema: list[db.TableInfo]) -> None:
    attempts = 0
    while True:
        check = validate_sql(st.session_state.sql_editor, table_names(schema))
        error, can_autofix = check.error, False
        if check.ok:
            try:
                with st.spinner("Running query…"):
                    result = db.execute_query(path, check.sql)
                st.session_state.update(result={"df": result.df, "truncated": result.truncated,
                                                "elapsed": result.elapsed, "sql": check.sql}, error=None)
                log_history(check.sql, True, len(result.df))
                suffix = f" after {attempts} automatic fix(es)" if attempts else ""
                return notify("success", f"Query returned {len(result.df):,} row(s){suffix}.")
            except db.QueryError as exc:
                error, can_autofix = str(exc), True
        if not (can_autofix and st.session_state.get("auto_fix") and attempts < MAX_AUTO_FIX and llm.is_configured()):
            st.session_state.update(result=None, error=error)
            log_history(st.session_state.sql_editor, False, error=error)
            return notify("error", "The query could not be run. See the details below.")
        attempts += 1
        try:
            with st.spinner(f"Query failed, asking the AI to fix it (attempt {attempts} of {MAX_AUTO_FIX})…"):
                st.session_state.sql_editor = llm.fix_sql(
                    st.session_state.active_question, db.schema_to_prompt(schema, st.session_state.active_question),
                    st.session_state.sql_editor, error)
        except llm.LLMError as exc:
            st.session_state.update(result=None, error=error)
            log_history(st.session_state.sql_editor, False, error=error)
            return notify("error", f"Query failed and automatic fix was unavailable: {exc}")


def do_fix(schema: list[db.TableInfo]) -> None:
    error = st.session_state.error
    if not error:
        return
    try:
        with st.spinner("Asking the AI to fix the SQL…"):
            fixed = llm.fix_sql(st.session_state.active_question, db.schema_to_prompt(schema, st.session_state.active_question),
                                st.session_state.sql_editor, error)
    except llm.LLMError as exc:
        return notify("error", str(exc))
    st.session_state.update(sql_editor=fixed, error=None, result=None)
    notify("info", "The AI suggested a fix. Review the SQL and run it again.")


def do_explain(schema: list[db.TableInfo]) -> None:
    check = validate_sql(st.session_state.sql_editor, table_names(schema))
    if not check.ok:
        return notify("warning", f"Fix the SQL before requesting an explanation: {check.error}")
    try:
        with st.spinner("Explaining the query…"):
            text = llm.explain_sql(check.sql, db.schema_to_prompt(schema, st.session_state.active_question))
    except llm.LLMError as exc:
        return notify("error", str(exc))
    st.session_state.explanation = {"sql": check.sql, "text": text}


# ------------------------------------------------------------ visualisation
def _looks_temporal(series: pd.Series, name: str) -> bool:
    if pd.api.types.is_datetime64_any_dtype(series):
        return True
    if pd.api.types.is_numeric_dtype(series):
        return False
    hinted = any(w in name.lower() for w in ("date", "month", "day", "week", "time"))
    parsed = pd.to_datetime(series.astype(str), errors="coerce", format="mixed")
    return bool(parsed.notna().all()) and (hinted or len(series) > 0)


def render_visualization(df: pd.DataFrame) -> None:
    numeric = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
    others = [c for c in df.columns if c not in numeric]
    if len(df) < 2 or len(df.columns) < 2 or not numeric:
        st.info("Charts need at least two rows, a label column and a numeric column. Try a query with GROUP BY.")
        return
    if len(df) > 500:
        st.info("This result has more than 500 rows. Add a LIMIT or aggregate it to chart it.")
        return
    key = abs(hash(tuple(df.columns)))
    x_options = others + [c for c in numeric]
    c1, c2, c3 = st.columns(3)
    x = c1.selectbox("X axis (labels)", x_options, key=f"viz_x_{key}")
    y_options = [c for c in numeric if c != x]
    if not y_options:
        st.info("Need a second numeric column for the Y axis.")
        return
    y = c2.selectbox("Y axis (values)", y_options, key=f"viz_y_{key}")

    temporal = _looks_temporal(df[x], x)
    kinds = ["Line chart", "Bar chart"] if temporal else ["Bar chart", "Line chart"]
    plot = pd.DataFrame({"x": df[x], "y": df[y]}).dropna()
    if len(plot) <= 12 and (plot["y"] >= 0).all():
        kinds.append("Pie chart")
    kind = c3.selectbox("Chart type", kinds, key=f"viz_k_{key}_{x}_{y}")
    st.caption(f"Suggested for this data: {kinds[0]}.")

    if temporal and kind != "Pie chart":
        plot["x"] = pd.to_datetime(plot["x"].astype(str), errors="coerce", format="mixed")
        x_enc = alt.X("x:T", title=x)
    elif kind == "Line chart":
        plot["x"] = plot["x"].astype(str)
        x_enc = alt.X("x:O", title=x, sort=None)
    else:
        plot["x"] = plot["x"].astype(str)
        x_enc = alt.X("x:N", title=x, sort=None)
    tip = [alt.Tooltip("x", title=x), alt.Tooltip("y:Q", title=y)]
    if kind == "Bar chart":
        chart = alt.Chart(plot).mark_bar(color=ACCENT).encode(x=x_enc, y=alt.Y("y:Q", title=y), tooltip=tip)
    elif kind == "Line chart":
        chart = alt.Chart(plot).mark_line(color=ACCENT, point=True).encode(x=x_enc, y=alt.Y("y:Q", title=y), tooltip=tip)
    else:
        chart = alt.Chart(plot).mark_arc(innerRadius=55).encode(
            theta=alt.Theta("y:Q"), color=alt.Color("x:N", title=x), tooltip=tip)
    st.altair_chart(chart.properties(height=380))


# ----------------------------------------------------------------- startup
_init_state()
try:
    db.ensure_sample_database()
except Exception as exc:  # noqa: BLE001
    st.error(f"Could not create the sample database: {exc}")
    st.stop()

databases = db.list_databases()
if st.session_state.get("db_name") not in databases:
    st.session_state.db_name = databases[0]

# ----------------------------------------------------------------- sidebar
with st.sidebar:
    st.markdown("### Database")
    st.selectbox("Active database", databases, key="db_name", on_change=reset_workspace,
                 label_visibility="collapsed")
    db_path = db.resolve_db_path(st.session_state.db_name)

    with st.expander("Add or create a database"):
        n = st.session_state.upload_n
        st.file_uploader("SQLite file", type=["db", "sqlite", "sqlite3"], key=f"db_upload_{n}")
        st.button("Add SQLite file", on_click=add_uploaded_db, key="btn_add_db")
        st.file_uploader("…or build one from a CSV", type=["csv"], key=f"csv_upload_{n}")
        st.button("Create from CSV", on_click=add_csv_db, key="btn_add_csv")
        st.button("Rebuild sample database", on_click=rebuild_sample, key="btn_rebuild")

    try:
        schema = db.get_schema(db_path)
    except db.DatabaseError as exc:
        st.error(str(exc))
        st.stop()

    st.markdown("### Schema")
    show_samples = st.checkbox("Show sample rows", value=False)
    if not schema:
        st.caption("This database has no tables.")
    for t in schema:
        with st.expander(f"{t.name}  ({t.row_count:,} rows)"):
            st.dataframe(pd.DataFrame([{"column": c.name, "type": c.type or "-",
                                        "key": "PK" if c.pk else "", "required": c.notnull} for c in t.columns]),
                         hide_index=True)
            for col, ref_t, ref_c in t.foreign_keys:
                st.caption(f"{col} references {ref_t}.{ref_c}")
            if show_samples:
                try:
                    st.dataframe(db.get_table_sample(db_path, t.name), hide_index=True)
                except db.DatabaseError as exc:
                    st.caption(str(exc))

    examples = example_questions(schema, st.session_state.db_name == db.SAMPLE_DB_NAME)
    if examples:
        st.markdown("### Example questions")
        for i, q in enumerate(examples):
            st.button(q, key=f"ex_{i}", on_click=use_example, args=(q,))

# -------------------------------------------------------------------- main
st.markdown('<p class="app-title">SQL Query Generator</p>'
            '<p class="app-sub">Ask a question in plain English. the AI writes the SQLite query, you review it, then run it.</p>',
            unsafe_allow_html=True)

if not llm.is_configured():
    st.warning(llm.SETUP_MESSAGE)
else:
    st.caption(f"AI provider: {llm.provider_label()}")

if not schema:
    st.markdown('<div class="empty">This database has no tables yet. Pick another database or upload one from the sidebar.</div>',
                unsafe_allow_html=True)
    st.stop()

st.text_area("Your question", key="question", height=90, max_chars=llm.MAX_QUESTION_CHARS,
             placeholder="e.g. Show the top 5 customers by total spending")
b1, b2 = st.columns([1, 3])
b1.button("Generate SQL", type="primary", on_click=request, args=("generate",), key="btn_generate")
b2.checkbox(f"Auto-fix failed queries with AI (up to {MAX_AUTO_FIX} tries)", value=True, key="auto_fix")

action = st.session_state.pop("pending", None)
if action == "generate":
    do_generate(schema)
elif action == "execute":
    do_execute(db_path, schema)
elif action == "fix":
    do_fix(schema)
elif action == "explain":
    do_explain(schema)

if st.session_state.notice:
    kind, text = st.session_state.notice
    {"success": st.success, "error": st.error, "warning": st.warning, "info": st.info}[kind](text)
    st.session_state.notice = None

st.subheader("SQL editor")
st.text_area("SQL", key="sql_editor", height=170, label_visibility="collapsed",
             placeholder="Generated SQL appears here. You can also write your own SELECT query.")
st.button("Execute query", type="primary", on_click=request, args=("execute",), key="btn_execute",
          disabled=not st.session_state.sql_editor.strip())
st.caption("Only a single read-only SELECT is allowed. The database is opened read-only with a time limit "
           f"of {db.DEFAULT_TIMEOUT:g}s and a {db.DEFAULT_MAX_ROWS:,}-row cap.")

tab_results, tab_chart, tab_explain, tab_history, tab_stats = st.tabs(
    ["Results", "Visualization", "Explanation", "History", "Database statistics"])

result = st.session_state.result
with tab_results:
    if st.session_state.error:
        st.error(st.session_state.error)
        st.button("Fix with AI", on_click=request, args=("fix",), key="btn_fix",
                  disabled=not llm.is_configured())
    elif result is None:
        st.markdown('<div class="empty">No results yet. Generate or write a query, then choose Execute query.</div>',
                    unsafe_allow_html=True)
    else:
        df: pd.DataFrame = result["df"]
        m1, m2, m3 = st.columns(3)
        m1.metric("Rows", f"{len(df):,}")
        m2.metric("Columns", len(df.columns))
        m3.metric("Run time", f"{result['elapsed'] * 1000:.0f} ms")
        if result["truncated"]:
            st.warning(f"Showing the first {len(df):,} rows. Add a LIMIT or filter to narrow the result.")
        if df.empty:
            st.info("The query ran but returned no rows.")
        else:
            term = st.text_input("Search results", key="result_filter", placeholder="Filter rows containing…")
            shown = df
            if term:
                mask = df.astype(str).apply(lambda c: c.str.contains(term, case=False, regex=False)).any(axis=1)
                shown = df[mask]
                st.caption(f"{len(shown):,} of {len(df):,} rows match.")
            st.dataframe(shown, hide_index=True)
            st.download_button("Download CSV", df.to_csv(index=False).encode("utf-8"),
                               file_name=f"query_results_{datetime.now():%Y%m%d_%H%M%S}.csv", mime="text/csv")

with tab_chart:
    if result is None or result["df"].empty:
        st.info("Run a query that returns rows to see chart suggestions.")
    else:
        render_visualization(result["df"])

with tab_explain:
    st.button("Explain this query", on_click=request, args=("explain",), key="btn_explain",
              disabled=not (llm.is_configured() and st.session_state.sql_editor.strip()))
    exp_state = st.session_state.explanation
    if exp_state:
        if exp_state["sql"] != validate_sql(st.session_state.sql_editor).sql:
            st.caption("The SQL changed since this explanation was written.")
        st.markdown(exp_state["text"])
    else:
        st.markdown('<div class="empty">Explains joins, filters, aggregations and sorting in plain language.</div>',
                    unsafe_allow_html=True)

with tab_history:
    history = st.session_state.history
    if not history:
        st.markdown('<div class="empty">Queries you run in this session will be listed here.</div>', unsafe_allow_html=True)
    else:
        if st.button("Clear history", key="btn_clear_history"):
            st.session_state.history = []
            st.rerun()
        for i, h in enumerate(history[:50]):
            label = f"{h['time']}  {'✅' if h['ok'] else '❌'}  {h['question'][:80]}"
            with st.expander(label):
                st.code(h["sql"], language="sql")
                st.caption(f"{h['rows']:,} row(s) returned" if h["ok"] else f"Failed: {h['error']}")
                st.button("Load into editor", key=f"reuse_{i}", on_click=reuse, args=(h,))

with tab_stats:
    stats = db.get_db_stats(db_path)
    s1, s2, s3, s4 = st.columns(4)
    s1.metric("Tables", stats["table_count"])
    s2.metric("Total rows", f"{stats['total_rows']:,}")
    s3.metric("Columns", stats["column_count"])
    s4.metric("File size", f"{stats['size_kb']:,} KB")
    st.dataframe(pd.DataFrame(stats["tables"]), hide_index=True)
    if stats["tables"]:
        st.bar_chart(pd.DataFrame(stats["tables"]).set_index("table")["rows"])
    st.caption(f"SQLite {stats['sqlite_version']}, views: {stats['view_count']}")
