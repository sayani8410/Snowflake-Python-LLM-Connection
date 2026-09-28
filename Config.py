
import os
from dotenv import load_dotenv
load_dotenv()

"""
Central Configuration. Loads everything from the .env file so no secrets are hardcoded in the pipeline scripts.
"""
SNOWFLAKE_CONFIG = {
    "user" : os.environ['SNOWFLAKE_USERNAME'],
    "password" : os.environ['SNOWFLAKE_PASSWORD'],
    "account" : os.environ['ACCOUNT_IDENTIFIER'],
    "warehouse" : os.environ['COMPUTE_WAREHOUSE'],
    "database" : os.environ['DATABASE'],
    "schema" : os.environ['SCHEMA'],
    "role" : os.environ['ROLE']
}

CHROMA_PERSIST_DIR = os.getenv("CHROMA_PERSIST_DIR", "./chroma_store")
EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"
CLAUDE_MODEL = "vertex_ai.anthropic.claude-opus-4-6"