# dpa_modulA.py
# Code-Zellen aus BE_AI_UC_DPA_modulA_PoC.ipynb

# --- Imports ---
import json
import os
from gen_ai_hub.proxy.native.openai import embeddings as native_embeddings
from hdbcli import dbapi
from gen_ai_hub.proxy.langchain.openai import ChatOpenAI, OpenAI
from gen_ai_hub.proxy.langchain.init_models import init_embedding_model
from langchain_community.vectorstores.hanavector import HanaDB
from langchain.text_splitter import CharacterTextSplitter
from langchain.document_loaders import PyPDFLoader
from langchain_experimental.text_splitter import SemanticChunker
from langchain.schema import Document

# --- Funktionen ---

def install_py_packages():
    """Dummy-Funktion für Package-Installation (nur Print)."""
    print("py-packages installed!")

# A0.2 load env-variables from config.json-file
def load_env_variables(config_file):
    """
    Load environment variables from a JSON configuration file.
    Args:
        config_file (str): Path to the JSON configuration file.
    Returns:
        dict: A dictionary containing the environment variables.
    """
    if not os.path.exists(config_file):
        raise FileNotFoundError(f"The configuration file {config_file} does not exist.")
    try:
        with open(config_file, 'r') as file:
            env_variables = json.load(file)
    except json.JSONDecodeError as e:
        raise ValueError(f"Error decoding JSON from the configuration file {config_file}: {e}")
    for key, value in env_variables.items():
        if isinstance(value, dict):
            value = json.dumps(value)
        os.environ[key] = str(value)
    return env_variables

def test_sap_ai_core_embedding():
    """Testet die Verbindung zu SAP AI Core Embedding-Modell."""
    model_embedding_name = os.getenv("AICORE_DEPLOYMENT_MODEL_EMBEDDING", "text-embedding-ada-002")
    try:
        response = native_embeddings.create(
            input="SAP Generative AI Hub is awesome!",
            model_name=model_embedding_name,
        )
        print(response.data)
    except ValueError as e:
        print(f"Error: {e}")
        print("Ensure the model name matches an existing deployment in SAP AI Hub.")

# A0.3 Setup and test connection to HANA DB
def setup_hana_connection():
    """Stellt eine Verbindung zur HANA DB her und gibt das Connection-Objekt zurück."""
    hdb_host_address = str(os.getenv("hdb_host_address"))
    hdb_user = str(os.getenv("hdb_user"))
    hdb_password = str(os.getenv("hdb_password"))
    hdb_port = int(os.getenv("hdb_port"))
    print(f"hdb_host_address: {hdb_host_address}")
    print(f"hdb_user: {hdb_user}")
    print(f"hdb_port: {hdb_port}")
    if not all([hdb_host_address, hdb_user, hdb_password, hdb_port]):
        raise ValueError("One or more HANA DB connection parameters are missing.")
    assert hdb_port is not None, "hdb_port must not be None"
    hana_connection = dbapi.connect(
        address=hdb_host_address,
        port=hdb_port,
        user=hdb_user,
        password=hdb_password,
        autocommit=True
    )
    return hana_connection

# A0.4 Setup LLM-Connection to SAP AI-HUB
def setup_llm():
    """Initialisiert das LLM über SAP AI-Hub."""
    aicore_model_name = str(os.getenv("AICORE_DEPLOYMENT_MODEL"))
    if not aicore_model_name:
        raise ValueError(f"Parameter LLM-Model-Name fehlt in der Umgebungskonfiguration.")
    llm = ChatOpenAI(proxy_model_name=aicore_model_name)
    if not llm:
        raise ValueError(f"Parameter LLM-Model-Name fehlt in der Umgebungskonfiguration.")
    else:
        print(f"Parameter LLM-Model-Name: {aicore_model_name} wurde erfolgreich geladen.")
    return llm

# A0.5 Setup embedding-model from AI Hub
def setup_embedding_model():
    """Initialisiert das Embedding-Modell über SAP AI-Hub."""
    ai_core_embedding_model_name = str(os.getenv('AICORE_DEPLOYMENT_MODEL_EMBEDDING'))
    try:
        embeddings = init_embedding_model(ai_core_embedding_model_name)
        print("Embedding model initialized successfully.")
        return embeddings
    except Exception as e:
        print("Embedding model not initialized.")
        print(e)
        return None

# A0.6 Setup vectorestore in SAP HANA Database
def setup_hana_vectorstore(embeddings, hana_connection):
    """Initialisiert den HanaDB VectorStore."""
    vector_table_name = str(os.getenv('hdb_table_name'))
    hana_database = HanaDB(
        embedding=embeddings,
        connection=hana_connection,
        table_name=vector_table_name
    )
    try:
        print(f"Successfully created SAP HANA VectorStore interface: {hana_database.connection} and table: {vector_table_name}.")
    except Exception as e:
        print(e)
    return hana_database

# function A2: load the pdf-file and split into text_chunks
def load_pdf(file_path):
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"The file {file_path} does not exist.")
    loader = PyPDFLoader(file_path)
    documents = loader.load()
    return documents

# function A2.2.3 split document in chunks - Semantic Chunker
def split_pdf_to_chunks(docs, chunk_size=1000, chunk_overlap=200):
    """Teilt Dokumente in Chunks auf."""
    chunker = CharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    chunks = []
    for doc in docs:
        for txt in chunker.split_text(doc.page_content):
            chunks.append(Document(page_content=txt, metadata=doc.metadata))
    return chunks

def semantic_chunking(documents, embeddings):
    """Splittet Dokumente semantisch in Chunks."""
    text_splitter = SemanticChunker(embeddings=embeddings, breakpoint_threshold_type="gradient")
    text_chunks = []
    for doc in documents:
        chunks = text_splitter.split_text(doc.page_content)
        for text in chunks:
            text_chunks.append(Document(page_content=text, metadata=doc.metadata))
    print(f"Generated {len(text_chunks)} chunks.")
    return text_chunks

# function A4.2.1 delete existing documents and load embeddings
def reload_embeddings(hana_database, text_chunks):
    hana_database.delete(filter={})
    hana_database.add_documents(text_chunks)
    print(f"Successfully added {len(text_chunks)} document chunks to the database.")
    print("Connected to the HANA Cloud database.")

# function A4.2 query to verify embeddings
def query_embeddings(hana_connection, hana_database, keyword="Rückstellung"):
    cursor = hana_connection.cursor()
    sql = f'SELECT VEC_TEXT, TO_NVARCHAR(VEC_VECTOR) FROM "{hana_database.table_name}" WHERE VEC_TEXT LIKE '%{keyword}%''
    cursor.execute(sql)
    vectors = cursor.fetchall()
    print(vectors[5:10])

# --- Ende ---
