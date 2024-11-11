import os, logging, sys
from sqlalchemy import create_engine
from llama_index.core import SQLDatabase
from llama_index.core.query_engine import NLSQLTableQueryEngine
from llama_index.core import Settings
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.llms.ollama import Ollama

logging.basicConfig(stream=sys.stdout, level=logging.DEBUG)
logging.getLogger().addHandler(logging.StreamHandler(stream=sys.stdout))

# bge-base embedding model
Settings.embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-base-en-v1.5")

# ollama
Settings.llm = Ollama(
  model="llama3", 
  temperature=0,
  max_tokens=1024,
  request_timeout=360.0,
  base_url="http://llm:11434",
)

uri = os.environ.get("DB_URL")
engine = create_engine(uri)

sql_database = SQLDatabase(engine)
tables = ['PESSOA','LOCALIZAPESSOA']

query_engine = NLSQLTableQueryEngine(
  sql_database=sql_database, tables=tables
)

query_str = "Quais os 20 mais antigos alunos da Universidade? (Utilize TOP ao invés de LIMIT na construção da query pois se trata de um banco SQL Server)"
response = query_engine.query(query_str)

print (response)