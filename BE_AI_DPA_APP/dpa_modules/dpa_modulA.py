# dpa_modulA.py
# Code-Zellen aus BE_AI_UC_DPA_modulA_PoC.ipynb

# A0.1 install py-packages
# (magic pip commands commented out for script compatibility)
# %pip install --upgrade pip
# %pip install --quiet hdbcli --break-system-packages
# %pip install --quiet generative-ai-hub-sdk[all] --break-system-packages
# %pip install --quiet folium --break-system-packages
# %pip install --quiet ipywidgets --break-system-packages
# %pip install --quiet pypdf
# %pip install --quiet -U ipykernel
# %pip install --quiet hana-ml
# %pip install --quiet sqlalchemy-hana
# %pip install --quiet nltk
# %pip install --quiet langchain langchain_experimental langchain_openai
print("py-packages installed!")

# A0.2 load env-variables from config.json-file
import json
import os

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

# Example usage
config_file = "/home/user/.aicore/config.json"
try:
    env_variables = load_env_variables(config_file)
    print(f"Loaded environment variables: {env_variables}")
except (FileNotFoundError, ValueError) as e:
    print(e)

# A0.2 Test connection with env-Variables to SAP AI core
from gen_ai_hub.proxy.native.openai import embeddings
import os

model_embedding_name = os.getenv("AICORE_DEPLOYMENT_MODEL_EMBEDDING", "text-embedding-ada-002")
try:
    response = embeddings.create(
        input="SAP Generative AI Hub is awesome!",
        model_name=model_embedding_name,
    )
    print(response.data)
except ValueError as e:
    print(f"Error: {e}")
    print("Ensure the model name matches an existing deployment in SAP AI Hub.")

# A0.3 Setup and test connection to HANA DB
import os
from hdbcli import dbapi

hdb_host_address = os.getenv("hdb_host_address")
hdb_user = os.getenv("hdb_user")
hdb_password = os.getenv("hdb_password")
hdb_port = os.getenv("hdb_port")

print(f"hdb_host_address: {hdb_host_address}")
print(f"hdb_user: {hdb_user}")
print(f"hdb_port: {hdb_port}")

if not all([hdb_host_address, hdb_user, hdb_password, hdb_port]):
    raise ValueError("One or more HANA DB connection parameters are missing.")

assert hdb_port is not None, "hdb_port must not be None"
hdb_port = int(hdb_port)

hana_connection = dbapi.connect(
    address=hdb_host_address,
    port=hdb_port,
    user=hdb_user,
    password=hdb_password,
    autocommit=True
)

# A0.4 Setup LLM-Connection to SAP AI-HUB
import os
import dotenv
from gen_ai_hub.proxy.langchain.openai import ChatOpenAI, OpenAI

aicore_model_name = str(os.getenv("AICORE_DEPLOYMENT_MODEL"))
if not aicore_model_name:
    raise ValueError(f"Parameter LLM-Model-Name fehlt in der Umgebungskonfiguration.")

llm = ChatOpenAI(proxy_model_name=aicore_model_name)
if not llm:
    raise ValueError(f"Parameter LLM-Model-Name fehlt in der Umgebungskonfiguration.")
else:
    print(f"Parameter LLM-Model-Name: {aicore_model_name} wurde erfolgreich geladen.")

# A0.5 Setup embedding-model from AI Hub
from gen_ai_hub.proxy.langchain.init_models import init_embedding_model

ai_core_embedding_model_name = str(os.getenv('AICORE_DEPLOYMENT_MODEL_EMBEDDING'))
try:
    embeddings = init_embedding_model(ai_core_embedding_model_name)
    print("Embedding model initialized successfully.")
except Exception as e:
    print("Embedding model not initialized.")
    print(e)

# A0.6 Setup vectorestore in SAP HANA Database
from langchain_community.vectorstores.hanavector import HanaDB

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

# function A2: load the pdf-file and split into text_chunks
import os
from langchain.text_splitter import CharacterTextSplitter
from langchain.document_loaders import PyPDFLoader

def load_pdf(file_path):
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"The file {file_path} does not exist.")
    loader = PyPDFLoader(file_path)
    documents = loader.load()
    return documents

if __name__ == "__main__":
    file_path = "data/sample_accounting_guide.pdf"
    try:
        documents = load_pdf(file_path)
        print(f"Length of text created: {len(documents)}")
        print(f"First page from Text: {documents[0]}")
    except FileNotFoundError as e:
        print(e)

# function A2.2.3 split document in chunks - Semantic Chunker
from langchain_experimental.text_splitter import SemanticChunker
from gen_ai_hub.proxy.langchain.init_models import init_embedding_model
from langchain.schema import Document

chunk_size_param = 1000
chunk_overlap_param = 200

documents = globals().get('documents', [])
ai_core_embedding_model_name = str(os.getenv('AICORE_DEPLOYMENT_MODEL_EMBEDDING'))
embeddings = init_embedding_model(ai_core_embedding_model_name)
text_splitter = SemanticChunker(embeddings=embeddings, breakpoint_threshold_type="gradient")

text_chunks = []
for doc in documents:
    chunks = text_splitter.split_text(doc.page_content)
    for text in chunks:
        text_chunks.append(Document(page_content=text, metadata=doc.metadata))

print(f"Generated {len(text_chunks)} chunks.")

# function A4.2.1 delete existing documents and load embeddings
hana_database.delete(filter={})
hana_database.add_documents(text_chunks)
print(f"Successfully added {len(text_chunks)} document chunks to the database.")
print("Connected to the HANA Cloud database.")

# function A4.2 query to verify embeddings
cursor = hana_connection.cursor()
sql = f'SELECT VEC_TEXT, TO_NVARCHAR(VEC_VECTOR) FROM "{hana_database.table_name}" WHERE VEC_TEXT LIKE \'%Rückstellung%\''
cursor.execute(sql)
vectors = cursor.fetchall()
print(vectors[5:10])
