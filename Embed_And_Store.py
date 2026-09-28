"""
STEP 2 - Generate Embeddings for each document.
STEP 3 - Store the text + Embedding + Metadata in the Vector Database (ChromaDB).
"""

from sentence_transformers import SentenceTransformer
import chromadb
from chromadb.utils import embedding_functions
from Config import EMBEDDING_MODEL_NAME, CHROMA_PERSIST_DIR, SNOWFLAKE_CONFIG
from Extraction import get_connection, list_tables, build_documents_for_table

COLLECTION_NAME = "snowflake_tables"

def get_chroma_collection():

    # This creates (or reopens) a Chroma database that saves to disk at whatever folder CHROMA_PERSIST_DIR points to (e.g., ./chroma_store). 
    # "Persistent" means the data survives after your script exits — unlike an in-memory client, which would lose everything once the program ends.
    client = chromadb.PersistentClient(path=CHROMA_PERSIST_DIR)

    # This wraps your embedding model (all-MiniLM-L6-v2) so Chroma can call it automatically whenever text needs to be turned into a vector 
    # — both when you add documents and later when you search.
    ef = embedding_functions.SentenceTransformerEmbeddingFunction(model_name=EMBEDDING_MODEL_NAME)
    
    # A "collection" in Chroma is like a table — a named bucket where your vectors + text + metadata live. 
    # get_or_create means: if a collection called "snowflake_tables" already exists on disk from a previous run, reuse it (and keep adding to it); otherwise, create a fresh one.
    collection = client.get_or_create_collection(name=COLLECTION_NAME, embedding_function=ef)
    
    return collection

# Setting the Embedding Model
def getEmbeddingModel() -> SentenceTransformer:
    return SentenceTransformer(EMBEDDING_MODEL_NAME)

# Converting the documents into the Embedding Vectors
def embed_and_store(documents: list[dict], batch_size: int = 256):
    """
    documents: list of {"text": str, "metadata": dict} as produced by Extraction.py file
    """

    collection = get_chroma_collection()

    # batch_size is the chunk size
    for i in range(0, len(documents), batch_size):

        batch = documents[i:i + batch_size]
        ids = [f"doc_{i + j}" for j in range(len(batch))]

        # Pulling out the text part - exactly what gets embedded — turned into vectors by the all-MiniLM-L6-v2 model.
        texts = [d["text"] for d in batch] 

        # Pulling out the metadata part
        metadatas = [d["metadata"] for d in batch]

        # Chroma computes embeddings internally via the embedding function and updates the collection
        collection.add(ids=ids, documents=texts, metadatas=metadatas)
        
        print(f"Stored batch {i} to {i + len(batch)}")

    print(f"Done. Collection now has {collection.count()} documents.")



if __name__ == "__main__":
    conn = get_connection()
    cfg = SNOWFLAKE_CONFIG
    tables = list_tables(conn, cfg['database'], cfg['schema'])

    all_docs = []

    pk_columns = {
        "BOOKINGS": "BOOKING_ID",
        "HOSTS": "HOST_ID",
        "LISTINGS": "LISTING_ID",
    }

    for t in tables:
        all_docs.extend(build_documents_for_table(conn, cfg['database'], 
                            cfg['schema'], t, pk_Column = pk_columns.get(t), row_limit = 3))
    
    conn.close()

    embed_and_store(all_docs)

    