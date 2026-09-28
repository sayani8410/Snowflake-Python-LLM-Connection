""" 
STEP 1 : Extract rows from the Snowflake tables and turn each row into a natural-language "document" that embeds well.

Flat JSON/CSV rows embed poorly from semantic search. 
A readable sentence built from the column names + values gives the embedding model much more useful signal.

A natural language document is a chunk of human-readable text, representing one unit of the data, phrased as real sentence 
instead of structured/coded data so that the embedding model can actually understand semantically.
"""

import snowflake.connector
import pandas as pd 
from Config import SNOWFLAKE_CONFIG

def get_connection():
    return snowflake.connector.connect(**SNOWFLAKE_CONFIG)

# Listing the tables of the database and the schema

def list_tables(conn, database: str, schema: str) -> list[str]:
    cursor = conn.cursor()
    cursor.execute(f"SHOW TABLES IN SCHEMA {database}.{schema}")
    rows = cursor.fetchall()

    cols = [col.name for col in cursor.description]

    name_idx = cols.index("name")
    return [r[name_idx] for r in rows]

# Fetching the table details and converting into a dataframe

def fetch_table(conn, database: str, schema: str, table: str, limit: int | None = None) -> pd.DataFrame:
    cursor = conn.cursor()
    query = f"SELECT * FROM {database}.{schema}.{table}"
    
    if limit:
        query += f" LIMIT {limit}"

    cursor.execute(query)
    columns = [col.name for col in cursor.description]

    rows = cursor.fetchall()
    return pd.DataFrame(rows, columns=columns)

# Row to Document Conversion Function

def row_to_document(table_name: str, row: pd.Series, pk_Column: str | None = None) -> dict:
    """
    Converts a single row into a natural-language sentence + metadata.
    Example output text:
       "Table ORDERS record — order_id: 1234, customer_name: John Doe,
       product: Widget A, quantity: 3, order_date: 2026-01-15, total: 450.00"
    """
    parts = [f"Table {table_name} record - "]
    parts += [f"{col}: {val}" for col,val in row.items()]
    text = " ".join(parts)

    metadata = {
        "table": table_name,
        "row_id": str(row[pk_Column]) if pk_Column and pk_Column in row else "",

    }

    return {"text": text, "metadata": metadata}

# Calls the fetch table function to generate the table rows
# and then the row_to_document function for document conversion

def build_documents_for_table(conn, database: str, schema: str, table: str, pk_Column: str | None = None, row_limit: int | None = None) -> list[dict]:

    df = fetch_table(conn, database, schema, table, limit=row_limit)
    return [row_to_document(table, row, pk_Column) for _, row in df.iterrows()]


if __name__ == "__main__":
    conn = get_connection()
    cfg = SNOWFLAKE_CONFIG
    tables = list_tables(conn, cfg['database'], cfg['schema'])
    print(f"Found {len(tables)} tables : {tables} \n")

    all_docs = []

    pk_columns = {
        "BOOKINGS": "BOOKING_ID",
        "HOSTS": "HOST_ID",
        "LISTINGS": "LISTING_ID",
    }


    for t in tables:

        # df = fetch_table(conn, cfg['database'], cfg['schema'], t, 10)
        # print(f"{t} Table")
        # print(df,"\n")

        docs = build_documents_for_table(conn, cfg['database'], cfg['schema'], t, pk_Column = pk_columns.get(t), row_limit = 3)
        print(f" {t}; {len(docs)} documents")
        print(docs)
        all_docs.extend(docs)

    print(f" Total documents built: {len(all_docs)}")
    conn.close()

    