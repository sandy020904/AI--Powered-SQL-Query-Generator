import pytest

from sql_validator import validate_sql

TABLES = ["customers", "orders", "order_items", "products"]


@pytest.mark.parametrize(
    "sql",
    [
        "SELECT * FROM customers",
        "SELECT * FROM customers;",
        "select c.first_name, count(*) from customers c join orders o on o.customer_id = c.customer_id group by 1",
        "WITH t AS (SELECT customer_id, COUNT(*) n FROM orders GROUP BY 1) SELECT * FROM t ORDER BY n DESC LIMIT 5",
        "SELECT name FROM products UNION SELECT first_name FROM customers",
        "SELECT strftime('%Y-%m', order_date) AS month, COUNT(*) FROM orders GROUP BY month",
        "SELECT * FROM customers WHERE customer_id IN (SELECT customer_id FROM orders)",
        "SELECT 'DROP TABLE x; --' AS text_value",
    ],
)
def test_valid_selects_are_accepted(sql):
    result = validate_sql(sql, TABLES)
    assert result.ok, result.error
    assert not result.sql.endswith(";")


@pytest.mark.parametrize(
    "sql",
    [
        "",
        "   ",
        "INSERT INTO customers (first_name) VALUES ('x')",
        "UPDATE products SET price = 0",
        "DELETE FROM orders",
        "DROP TABLE customers",
        "ALTER TABLE customers ADD COLUMN x TEXT",
        "CREATE TABLE t (id INTEGER)",
        "ATTACH DATABASE 'other.db' AS other",
        "DETACH DATABASE other",
        "PRAGMA table_info(customers)",
        "SELECT 1; DROP TABLE customers",
        "SELECT * FROM customers; SELECT * FROM orders",
        "SELECT load_extension('evil')",
        "SELECT readfile('/etc/passwd')",
        "SELECT * FROM sqlite_master",
        "SELECT * FROM other.customers",
        "WITH x AS (SELECT 1) DELETE FROM customers",
        "this is not sql at all",
        "VACUUM",
    ],
)
def test_unsafe_or_invalid_sql_is_rejected(sql):
    result = validate_sql(sql, TABLES)
    assert not result.ok
    assert result.error


def test_unknown_table_is_rejected_when_schema_given():
    result = validate_sql("SELECT * FROM missing_table", TABLES)
    assert not result.ok
    assert "missing_table" in result.error


def test_cte_name_is_not_treated_as_unknown_table():
    assert validate_sql("WITH recent AS (SELECT * FROM orders) SELECT * FROM recent", TABLES).ok


def test_table_check_is_skipped_without_schema():
    assert validate_sql("SELECT * FROM anything").ok


def test_overlong_sql_is_rejected():
    assert not validate_sql("SELECT " + "1," * 6000 + "1").ok
