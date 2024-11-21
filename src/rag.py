from llama_index.core import (
  VectorStoreIndex, 
  SimpleDirectoryReader, 
  Settings,
  StorageContext,
  load_index_from_storage,
)
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.llms.ollama import Ollama
import logging
import sys, os, time

def debug():
  logging.basicConfig(stream=sys.stdout, level=logging.DEBUG)
  logging.getLogger().addHandler(logging.StreamHandler(stream=sys.stdout))

def load_data():
  return SimpleDirectoryReader("rag").load_data()

def set_embedding():
  print("-- Setting embedding model...")
  # bge-base embedding model
  Settings.embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-base-en-v1.5")

  print("-- Setting llm model...")
  # ollama
  Settings.llm = Ollama(
    model="llama3.2", 
    temperature=0,
    max_tokens=1024,
    request_timeout=360.0,
    base_url="http://llm:11434",
  )

def storage_data():
  PERSIST_DIR = "./storage"
  print("- Setting embedding and llm model...")
  set_embedding()
  
  if not os.path.exists(PERSIST_DIR):
    # load the documents and create the index
    documents = load_data()
    # prepare model embedding
    index = VectorStoreIndex.from_documents(documents)
    # store it for later
    index.storage_context.persist(persist_dir=PERSIST_DIR)
  else:
    # load the existing index
    storage_context = StorageContext.from_defaults(persist_dir=PERSIST_DIR)
    index = load_index_from_storage(storage_context)

  return index

if __name__ == "__main__":  
  #debug()
  print("\nRecover storage data...")
  index = storage_data()
  
  print("\nSet chat engine...")
  chat_engine = index.as_chat_engine(
    similarity_top_k=3,
    chat_mode="condense_question", 
    streaming=True,
  )

  while True:
    # Prompt user for input
    message = input("\nUser: ")

    start_time = time.time()
    # Exit program if user inputs "quit"
    if message.lower() == "quit":
      break

    response_stream = chat_engine.stream_chat(message)
    response_stream.print_response_stream()
    end_time = time.time()
    
    print(f"\nTime: {end_time - start_time} seconds")

  