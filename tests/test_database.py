import pytest

import database as db


@pytest.fixture()
def sample(tmp_path, monkeypatch):
    monkeypatch.setattr(db, "DATA_DIR", tmp_path.resolve())
    return db.ensure_sample_database()


def test_sample_database_is_created_with_expected_tables(sample):
    schema = {t.name: t for t in db.get_schema(sample)}
    assert set(schema) == {"customers", "products", "orders", "order_items"}
    assert all(t.row_count > 0 for t in schema.values())
    assert any(ref == "customers" for _c, ref, _r in schema["orders"].foreign_keys)


def test_sample_database_is_listed_first(sample):
    assert db.list_databases() == [db.SAMPLE_DB_NAME]


def test_schema_prompt_has_structure_but_no_rows(sample):
    text = db.schema_to_prompt(db.get_schema(sample))
    assert "TABLE customers" in text and "FOREIGN KEY" in text
    assert "@example.com" not in text  # no row data leaks into the prompt


def test_execute_select_returns_dataframe(sample):
    result = db.execute_query(sample, "SELECT category, COUNT(*) AS n FROM products GROUP BY category")
    assert list(result.df.columns) == ["category", "n"]
    assert result.df["n"].sum() == 25
    assert not result.truncated


def test_row_limit_truncates(sample):
    result = db.execute_query(sample, "SELECT * FROM order_items", max_rows=10)
    assert len(result.df) == 10 and result.truncated


@pytest.mark.parametrize(
    "sql",
    ["DELETE FROM customers", "DROP TABLE orders", "PRAGMA table_info(orders)", "SELECT 1; SELECT 2"],
)
def test_connection_layer_blocks_writes_even_without_validator(sample, sql):
    with pytest.raises(db.QueryError):
        db.execute_query(sample, sql)
    assert db.get_db_stats(sample)["table_count"] == 4


def test_invalid_sql_raises_query_error(sample):
    with pytest.raises(db.QueryError):
        db.execute_query(sample, "SELECT * FROM nope")


def test_runaway_query_times_out(sample):
    sql = "WITH RECURSIVE c(x) AS (SELECT 1 UNION ALL SELECT x + 1 FROM c) SELECT COUNT(*) FROM c"
    with pytest.raises(db.QueryError, match="longer than"):
        db.execute_query(sample, sql, timeout=0.5)


@pytest.mark.parametrize("name", ["../evil.db", "a/b.db", "..\\evil.db", "notes.txt", "", "C:\\x.db"])
def test_path_traversal_and_bad_names_rejected(sample, name):
    with pytest.raises(db.DatabaseError):
        db.resolve_db_path(name)


def test_upload_rejects_non_sqlite_data(sample):
    with pytest.raises(db.DatabaseError):
        db.save_uploaded_database("x.db", b"definitely not sqlite")


def test_upload_accepts_valid_database_and_avoids_overwrite(sample):
    data = sample.read_bytes()
    first = db.save_uploaded_database(db.SAMPLE_DB_NAME, data)
    assert first != db.SAMPLE_DB_NAME
    assert first in db.list_databases()


def test_csv_import_creates_queryable_database(sample):
    name = db.create_database_from_csv("my data.csv", b"city,sales\nAustin,10\nBerlin,20\n")
    result = db.execute_query(db.resolve_db_path(name), "SELECT SUM(sales) AS total FROM my_data")
    assert result.df["total"].iloc[0] == 30


def test_empty_csv_is_rejected(sample):
    with pytest.raises(db.DatabaseError):
        db.create_database_from_csv("empty.csv", b"a,b\n")
