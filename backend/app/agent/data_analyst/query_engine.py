
from sqlalchemy import text
from sqlalchemy.orm import Session


def execute_query(db: Session, query: str, params: dict | None = None):
    """
    Execute a SQL query and return the results.
    """

    result = db.execute(
        text(query),
        params or {},
    )

    return [dict(row) for row in result.mappings().all()]


def get_total_revenue(db: Session):
    query = """
        SELECT COALESCE(SUM(total_amount), 0) AS total_revenue
        FROM sales
    """

    return execute_query(db, query)


def get_top_selling_products(db: Session, limit: int = 5):
    query = """
        SELECT
            p.id,
            p.name,
            SUM(s.quantity) AS total_sold,
            SUM(s.total_amount) AS total_revenue
        FROM sales s
        JOIN products p ON p.id = s.product_id
        GROUP BY p.id, p.name
        ORDER BY total_sold DESC
        LIMIT :limit
    """

    return execute_query(db, query, {"limit": limit})


def get_low_stock_products(db: Session):
    query = """
        SELECT
            p.id AS product_id,
            p.name,
            i.quantity,
            i.reorder_level,
            i.target_stock
        FROM inventory i
        JOIN products p ON p.id = i.product_id
        WHERE i.quantity <= i.reorder_level
        ORDER BY i.quantity ASC
    """

    return execute_query(db, query)
