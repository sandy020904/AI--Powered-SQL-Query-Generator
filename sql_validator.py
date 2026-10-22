"""Static validation of SQL before it is allowed anywhere near the database.

Only a single read-only SELECT statement (optionally with CTEs / set operations)
is accepted. This is one layer of defence: database.py additionally opens the
database read-only and installs a SQLite authorizer that denies everything
except reads.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import sqlglot
from sqlglot import exp
from sqlglot.errors import SqlglotError

MAX_SQL_LENGTH = 10_000


def _classes(*names: str) -> tuple[type, ...]:
    """Collect sqlglot expression classes that exist in the installed version."""
    return tuple(getattr(exp, n) for n in names if hasattr(exp, n))


ALLOWED_ROOTS = _classes("Select", "Union", "Intersect", "Except", "Subquery")

FORBIDDEN_NODES = _classes(
    "Insert", "Update", "Delete", "Drop", "Alter", "AlterTable", "Create", "Command",
    "Pragma", "Attach", "Detach", "Merge", "Transaction", "Commit", "Rollback",
    "Set", "Use", "Copy", "Analyze", "TruncateTable", "Into", "LoadData",
)

# Functions that read/write files, load code, or can exhaust memory.
BLOCKED_FUNCTIONS = frozenset(
    {
        "load_extension", "readfile", "writefile", "edit", "fts3_tokenizer",
        "zeroblob", "randomblob", "sqlite_compileoption_get", "sqlite_source_id",
    }
)


@dataclass(frozen=True)
class ValidationResult:
    ok: bool
    sql: str = ""
    error: str = ""


def _fail(message: str) -> ValidationResult:
    return ValidationResult(ok=False, error=message)


def validate_sql(sql: str, allowed_tables: Iterable[str] | None = None) -> ValidationResult:
    """Validate ``sql``. On success ``result.sql`` is the cleaned statement."""
    if not isinstance(sql, str) or not sql.strip():
        return _fail("The SQL is empty.")

    cleaned = sql.strip()
    while cleaned.endswith(";"):
        cleaned = cleaned[:-1].rstrip()
    if not cleaned:
        return _fail("The SQL is empty.")
    if len(cleaned) > MAX_SQL_LENGTH:
        return _fail(f"The SQL is too long (limit {MAX_SQL_LENGTH:,} characters).")

    try:
        statements = [s for s in sqlglot.parse(cleaned, read="sqlite") if s is not None]
    except SqlglotError as exc:
        first_line = str(exc).splitlines()[0] if str(exc) else "syntax error"
        return _fail(f"Could not parse the SQL: {first_line}")
    except Exception:  # noqa: BLE001 - parser must never crash the app
        return _fail("Could not parse the SQL.")

    if len(statements) != 1:
        return _fail("Only a single SQL statement is allowed.")

    tree = statements[0]
    if not isinstance(tree, ALLOWED_ROOTS):
        kind = type(tree).__name__.upper()
        return _fail(f"Only read-only SELECT queries are allowed (found {kind}).")

    allowed = {t.lower() for t in allowed_tables} if allowed_tables is not None else None
    cte_names = {cte.alias_or_name.lower() for cte in tree.find_all(exp.CTE)}

    for node in tree.find_all(exp.Expression):
        if isinstance(node, FORBIDDEN_NODES):
            return _fail(f"{type(node).__name__.upper()} operations are not allowed.")

        if isinstance(node, exp.Func):
            name = node.name if isinstance(node, exp.Anonymous) else node.sql_name()
            if (name or "").lower() in BLOCKED_FUNCTIONS:
                return _fail(f"The function {name}() is not allowed.")

        if isinstance(node, exp.Table):
            if isinstance(node.this, exp.Func):
                continue  # table-valued function; function blocklist already applied
            name = (node.name or "").lower()
            if not name:
                continue
            if node.db and node.db.lower() != "main":
                return _fail("Referencing other databases or schemas is not allowed.")
            if name.startswith("sqlite_"):
                return _fail("Access to SQLite internal tables is not allowed.")
            if allowed is not None and name not in allowed and name not in cte_names:
                return _fail(f"Table '{node.name}' does not exist in this database.")

    return ValidationResult(ok=True, sql=cleaned)
