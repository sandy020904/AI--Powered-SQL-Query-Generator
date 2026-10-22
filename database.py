"""SQLite access layer: sample data, uploads, schema inspection, safe execution."""
from __future__ import annotations

import io
import os
import random
import re
import sqlite3
import time
import uuid
from dataclasses import dataclass, field
from datetime import date, timedelta
from pathlib import Path

import pandas as pd


def _env_number(name: str, default: float, cast=float):
    try:
        return cast(os.getenv(name, default))
    except (TypeError, ValueError):
        return default


BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = Path(os.getenv("SQLGEN_DATA_DIR") or BASE_DIR / "data").resolve()
SAMPLE_DB_NAME = "sample_store.db"
MAX_UPLOAD_MB = _env_number("MAX_UPLOAD_MB", 50, int)
DEFAULT_MAX_ROWS = _env_number("MAX_RESULT_ROWS", 1000, int)
DEFAULT_TIMEOUT = _env_number("QUERY_TIMEOUT_SECONDS", 10.0, float)

SQLITE_HEADER = b"SQLite format 3\x00"
_NAME_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_\-. ]{0,100}\.(db|sqlite|sqlite3)$", re.IGNORECASE)

# SQLite authorizer action codes (stable across SQLite versions).
_SQLITE_READ, _SQLITE_SELECT, _SQLITE_FUNCTION, _SQLITE_RECURSIVE = 20, 21, 31, 33
_DENIED_FUNCTIONS = frozenset(
    {"load_extension", "readfile", "writefile", "edit", "fts3_tokenizer", "zeroblob", "randomblob"}
)


class DatabaseError(Exception):
    """Problem with a database file (missing, malformed, not allowed)."""


class QueryError(Exception):
    """A query failed to execute. The message is safe to show to users."""


@dataclass
class ColumnInfo:
    name: str
    type: str
    notnull: bool
    pk: bool


@dataclass
class TableInfo:
    name: str
    kind: str  # "table" or "view"
    columns: list[ColumnInfo] = field(default_factory=list)
    foreign_keys: list[tuple[str, str, str]] = field(default_factory=list)  # (column, ref_table, ref_column)
    row_count: int = 0


@dataclass
class QueryResult:
    df: pd.DataFrame
    truncated: bool
    elapsed: float


# --------------------------------------------------------------------------- paths
def ensure_data_dir() -> Path:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    return DATA_DIR


def resolve_db_path(name: str) -> Path:
    """Map a database file name to a path guaranteed to live inside DATA_DIR."""
    if not name or not _NAME_RE.match(name):
        raise DatabaseError("Invalid database name. Use letters, numbers, spaces, '-', '_' and a .db/.sqlite extension.")
    ensure_data_dir()
    path = (DATA_DIR / name).resolve()
    if path.parent != DATA_DIR:
        raise DatabaseError("Database path is outside the allowed data directory.")
    return path


def list_databases() -> list[str]:
    ensure_data_dir()
    names = [p.name for p in DATA_DIR.iterdir() if p.is_file() and _NAME_RE.match(p.name)]
    return sorted(names, key=lambda n: (n != SAMPLE_DB_NAME, n.lower()))


def _unique_name(name: str) -> str:
    stem, suffix = Path(name).stem, Path(name).suffix
    candidate, i = name, 1
    while (DATA_DIR / candidate).exists():
        i += 1
        candidate = f"{stem}_{i}{suffix}"
    return candidate


def quote_ident(identifier: str) -> str:
    return '"' + identifier.replace('"', '""') + '"'


# ----------------------------------------------------------------- connections
def _authorizer(action: int, arg1, arg2, dbname, source) -> int:
    if action == _SQLITE_FUNCTION:
        return sqlite3.SQLITE_DENY if (arg2 or "").lower() in _DENIED_FUNCTIONS else sqlite3.SQLITE_OK
    if action in (_SQLITE_READ, _SQLITE_SELECT, _SQLITE_RECURSIVE):
        return sqlite3.SQLITE_OK
    return sqlite3.SQLITE_DENY


def connect_readonly(path: Path, restricted: bool = False) -> sqlite3.Connection:
    """Open ``path`` read-only. ``restricted`` also installs the SELECT-only authorizer."""
    if not path.is_file():
        raise DatabaseError(f"Database file not found: {path.name}")
    try:
        conn = sqlite3.connect(f"{path.as_uri()}?mode=ro", uri=True, timeout=5)
        if restricted:
            conn.set_authorizer(_authorizer)
        return conn
    except sqlite3.Error as exc:
        raise DatabaseError(f"Could not open database: {exc}") from exc


# ---------------------------------------------------------------------- schema
def get_schema(path: Path) -> list[TableInfo]:
    conn = connect_readonly(path)
    try:
        rows = conn.execute(
            "SELECT name, type FROM sqlite_master WHERE type IN ('table','view') "
            "AND name NOT LIKE 'sqlite_%' ORDER BY type, name"
        ).fetchall()
        tables: list[TableInfo] = []
        for name, kind in rows:
            info = TableInfo(name=name, kind=kind)
            for _cid, col, ctype, notnull, _dflt, pk in conn.execute(
                "SELECT * FROM pragma_table_info(?)", (name,)
            ):
                info.columns.append(ColumnInfo(col, ctype or "", bool(notnull), bool(pk)))
            for fk in conn.execute("SELECT * FROM pragma_foreign_key_list(?)", (name,)):
                info.foreign_keys.append((fk[3], fk[2], fk[4] or ""))
            try:
                info.row_count = conn.execute(f"SELECT COUNT(*) FROM {quote_ident(name)}").fetchone()[0]
            except sqlite3.Error:
                info.row_count = 0
            tables.append(info)
        return tables
    except sqlite3.DatabaseError as exc:
        raise DatabaseError(f"This file is not a readable SQLite database: {exc}") from exc
    finally:
        conn.close()


def schema_to_prompt(schema: list[TableInfo], question: str | None = None, max_tables: int = 25) -> str:
    """Render schema metadata (no row data) as compact text for the LLM."""
    selected = schema
    if len(schema) > max_tables and question:
        words = set(re.findall(r"[a-z0-9]+", question.lower()))

        def score(t: TableInfo) -> int:
            names = [t.name.lower()] + [c.name.lower() for c in t.columns]
            return sum(1 for n in names for w in words if len(w) > 2 and (w in n or n in w))

        ranked = sorted(schema, key=score, reverse=True)[:max_tables]
        keep = {t.name for t in ranked}
        for t in ranked:  # pull in tables referenced by foreign keys
            keep.update(ref for _c, ref, _r in t.foreign_keys)
        selected = [t for t in schema if t.name in keep]

    lines: list[str] = []
    for t in selected:
        lines.append(f"{t.kind.upper()} {t.name}")
        for c in t.columns:
            extras = (" PRIMARY KEY" if c.pk else "") + (" NOT NULL" if c.notnull else "")
            lines.append(f"  - {c.name} {c.type or 'ANY'}{extras}")
        for col, ref_t, ref_c in t.foreign_keys:
            lines.append(f"  FOREIGN KEY {col} -> {ref_t}({ref_c})")
        lines.append("")
    return "\n".join(lines).strip()


def get_table_sample(path: Path, table: str, limit: int = 5) -> pd.DataFrame:
    """Sample rows for display to the *user* only; never sent to the LLM."""
    if table not in {t.name for t in get_schema(path)}:
        raise DatabaseError(f"Unknown table: {table}")
    conn = connect_readonly(path)
    try:
        return _to_frame(conn.execute(f"SELECT * FROM {quote_ident(table)} LIMIT ?", (limit,)))[0]
    except sqlite3.Error as exc:
        raise DatabaseError(str(exc)) from exc
    finally:
        conn.close()


def get_db_stats(path: Path) -> dict:
    schema = get_schema(path)
    tables = [t for t in schema if t.kind == "table"]
    return {
        "table_count": len(tables),
        "view_count": len(schema) - len(tables),
        "total_rows": sum(t.row_count for t in tables),
        "column_count": sum(len(t.columns) for t in schema),
        "size_kb": round(path.stat().st_size / 1024, 1),
        "sqlite_version": sqlite3.sqlite_version,
        "tables": [{"table": t.name, "type": t.kind, "rows": t.row_count, "columns": len(t.columns)} for t in schema],
    }


# ------------------------------------------------------------------- execution
def _to_frame(cursor: sqlite3.Cursor, max_rows: int | None = None) -> tuple[pd.DataFrame, bool]:
    rows = cursor.fetchmany(max_rows + 1) if max_rows is not None else cursor.fetchall()
    if cursor.description is None:
        raise QueryError("The query did not return any columns.")
    truncated = max_rows is not None and len(rows) > max_rows
    if truncated:
        rows = rows[:max_rows]
    seen: dict[str, int] = {}
    columns: list[str] = []
    for d in cursor.description:  # make column names unique for display
        n = str(d[0])
        seen[n] = seen.get(n, 0) + 1
        columns.append(n if seen[n] == 1 else f"{n}_{seen[n]}")
    clean = [[f"<{len(v)} bytes>" if isinstance(v, (bytes, bytearray)) else v for v in row] for row in rows]
    return pd.DataFrame(clean, columns=columns), truncated


def execute_query(
    path: Path, sql: str, max_rows: int = DEFAULT_MAX_ROWS, timeout: float = DEFAULT_TIMEOUT
) -> QueryResult:
    """Run an already-validated query on a read-only, authorizer-restricted connection."""
    conn = connect_readonly(path, restricted=True)
    deadline = time.monotonic() + timeout
    conn.set_progress_handler(lambda: 1 if time.monotonic() > deadline else 0, 10_000)
    start = time.monotonic()
    try:
        df, truncated = _to_frame(conn.execute(sql), max_rows)
    except sqlite3.OperationalError as exc:
        msg = str(exc)
        if "interrupted" in msg.lower():
            raise QueryError(f"The query took longer than {timeout:g}s and was stopped.") from exc
        raise QueryError(msg) from exc
    except sqlite3.Warning as exc:  # e.g. "You can only execute one statement at a time"
        raise QueryError(str(exc)) from exc
    except sqlite3.DatabaseError as exc:
        raise QueryError(f"Database error: {exc}") from exc
    finally:
        conn.close()
    return QueryResult(df=df, truncated=truncated, elapsed=time.monotonic() - start)


# --------------------------------------------------------------- create/upload
def save_uploaded_database(filename: str, data: bytes) -> str:
    """Validate and store an uploaded SQLite file. Returns the stored file name."""
    ensure_data_dir()
    if len(data) > MAX_UPLOAD_MB * 1024 * 1024:
        raise DatabaseError(f"File is larger than the {MAX_UPLOAD_MB} MB limit.")
    if not data.startswith(SQLITE_HEADER):
        raise DatabaseError("This is not a SQLite 3 database file.")
    safe = re.sub(r"[^A-Za-z0-9_\-. ]", "_", Path(filename).name).lstrip("._ ")
    if not _NAME_RE.match(safe):
        raise DatabaseError("Please upload a file ending in .db, .sqlite or .sqlite3.")
    final = DATA_DIR / _unique_name(safe)
    tmp = DATA_DIR / f".upload_{uuid.uuid4().hex}.tmp"
    try:
        tmp.write_bytes(data)
        get_schema(tmp)  # raises DatabaseError if malformed
        os.replace(tmp, final)
    finally:
        tmp.unlink(missing_ok=True)
    return final.name


def create_database_from_csv(filename: str, data: bytes) -> str:
    """Create a new one-table database from an uploaded CSV file."""
    ensure_data_dir()
    if len(data) > MAX_UPLOAD_MB * 1024 * 1024:
        raise DatabaseError(f"File is larger than the {MAX_UPLOAD_MB} MB limit.")
    try:
        df = pd.read_csv(io.BytesIO(data))
    except Exception as exc:  # noqa: BLE001
        raise DatabaseError(f"Could not read the CSV file: {exc}") from exc
    if df.empty or len(df.columns) == 0:
        raise DatabaseError("The CSV file has no data.")
    df.columns = [re.sub(r"\W+", "_", str(c).strip()).strip("_") or f"column_{i}" for i, c in enumerate(df.columns, 1)]
    df = df.loc[:, ~df.columns.duplicated()]
    stem = re.sub(r"[^A-Za-z0-9_]", "_", Path(filename).stem).strip("_") or "data"
    if stem[0].isdigit():
        stem = f"t_{stem}"
    final = DATA_DIR / _unique_name(f"{stem}.db")
    conn = sqlite3.connect(final)
    try:
        df.to_sql(stem, conn, index=False)
        conn.commit()
    except Exception as exc:  # noqa: BLE001
        conn.close()
        final.unlink(missing_ok=True)
        raise DatabaseError(f"Could not create the database: {exc}") from exc
    conn.close()
    return final.name


# ---------------------------------------------------------------- sample data
_FIRST = ["Ava", "Liam", "Noah", "Emma", "Olivia", "Arjun", "Priya", "Mateo", "Sofia", "Chen", "Mei", "Lucas",
          "Isla", "Omar", "Fatima", "Hugo", "Zoe", "Ravi", "Anika", "Ethan"]
_LAST = ["Smith", "Johnson", "Patel", "Garcia", "Nguyen", "Kim", "Müller", "Silva", "Brown", "Reddy",
         "Khan", "Rossi", "Tanaka", "Okafor", "Lopez", "Wilson"]
_PLACES = [("Austin", "USA"), ("Seattle", "USA"), ("Chicago", "USA"), ("Toronto", "Canada"), ("Vancouver", "Canada"),
           ("London", "UK"), ("Berlin", "Germany"), ("Lisbon", "Portugal"), ("Mumbai", "India"),
           ("Bengaluru", "India"), ("Sydney", "Australia"), ("Singapore", "Singapore")]
_PRODUCTS = [
    ("Wireless Earbuds", "Electronics", 79.99), ("Mechanical Keyboard", "Electronics", 119.00),
    ("27-inch Monitor", "Electronics", 289.50), ("USB-C Hub", "Electronics", 39.90),
    ("Portable Charger", "Electronics", 34.99), ("HD Webcam", "Electronics", 64.00),
    ("Ceramic Mug Set", "Home", 24.50), ("Desk Lamp", "Home", 42.00), ("Throw Blanket", "Home", 38.75),
    ("Scented Candle", "Home", 18.00), ("Cast Iron Skillet", "Home", 54.90), ("French Press", "Home", 32.00),
    ("Notebook Pack", "Office", 12.99), ("Desk Mat", "Office", 49.00), ("Gel Pen Set", "Office", 9.50),
    ("Desk Organizer", "Office", 27.25), ("Ergonomic Chair", "Office", 249.00),
    ("Yoga Mat", "Sports", 29.99), ("Resistance Bands", "Sports", 19.95), ("Water Bottle", "Sports", 22.00),
    ("Running Belt", "Sports", 17.50), ("Foam Roller", "Sports", 26.00),
    ("SQL Fundamentals", "Books", 34.00), ("Data Storytelling", "Books", 29.50), ("Python Cookbook", "Books", 44.00),
]

_SAMPLE_SCHEMA = """
CREATE TABLE customers (
    customer_id INTEGER PRIMARY KEY,
    first_name TEXT NOT NULL,
    last_name TEXT NOT NULL,
    email TEXT NOT NULL UNIQUE,
    city TEXT,
    country TEXT,
    signup_date TEXT NOT NULL
);
CREATE TABLE products (
    product_id INTEGER PRIMARY KEY,
    name TEXT NOT NULL,
    category TEXT NOT NULL,
    price REAL NOT NULL,
    stock_quantity INTEGER NOT NULL DEFAULT 0
);
CREATE TABLE orders (
    order_id INTEGER PRIMARY KEY,
    customer_id INTEGER NOT NULL REFERENCES customers(customer_id),
    order_date TEXT NOT NULL,
    status TEXT NOT NULL
);
CREATE TABLE order_items (
    order_item_id INTEGER PRIMARY KEY,
    order_id INTEGER NOT NULL REFERENCES orders(order_id),
    product_id INTEGER NOT NULL REFERENCES products(product_id),
    quantity INTEGER NOT NULL,
    unit_price REAL NOT NULL
);
CREATE INDEX idx_orders_customer ON orders(customer_id);
CREATE INDEX idx_items_order ON order_items(order_id);
CREATE INDEX idx_items_product ON order_items(product_id);
"""


def ensure_sample_database(overwrite: bool = False) -> Path:
    """Create the sample store database if it does not exist yet."""
    ensure_data_dir()
    path = DATA_DIR / SAMPLE_DB_NAME
    if path.exists() and not overwrite:
        return path
    rng = random.Random(42)
    tmp = DATA_DIR / f".sample_{uuid.uuid4().hex}.tmp"
    conn = sqlite3.connect(tmp)
    try:
        conn.executescript(_SAMPLE_SCHEMA)
        end = date(2025, 12, 31)
        customers = []
        for i in range(1, 41):
            first, last = rng.choice(_FIRST), rng.choice(_LAST)
            city, country = rng.choice(_PLACES)
            signup = date(2023, 6, 1) + timedelta(days=rng.randint(0, 560))
            customers.append((i, first, last, f"{first}.{last}{i}@example.com".lower(), city, country, signup.isoformat()))
        conn.executemany("INSERT INTO customers VALUES (?,?,?,?,?,?,?)", customers)
        conn.executemany(
            "INSERT INTO products VALUES (?,?,?,?,?)",
            [(i, n, c, p, rng.randint(0, 200)) for i, (n, c, p) in enumerate(_PRODUCTS, 1)],
        )
        weights = [max(1, 40 - i) for i in range(40)]  # some customers order far more often
        statuses = ["delivered", "shipped", "pending", "cancelled", "returned"]
        status_w = [70, 12, 8, 7, 3]
        item_id = 0
        for order_id in range(1, 161):
            cust = rng.choices(customers, weights=weights)[0]
            signup = date.fromisoformat(cust[6])
            odate = signup + timedelta(days=rng.randint(0, (end - signup).days))
            conn.execute(
                "INSERT INTO orders VALUES (?,?,?,?)",
                (order_id, cust[0], odate.isoformat(), rng.choices(statuses, status_w)[0]),
            )
            for pid in rng.sample(range(1, len(_PRODUCTS) + 1), rng.randint(1, 4)):
                item_id += 1
                conn.execute(
                    "INSERT INTO order_items VALUES (?,?,?,?,?)",
                    (item_id, order_id, pid, rng.randint(1, 3), _PRODUCTS[pid - 1][2]),
                )
        conn.commit()
    finally:
        conn.close()
    os.replace(tmp, path)
    return path
