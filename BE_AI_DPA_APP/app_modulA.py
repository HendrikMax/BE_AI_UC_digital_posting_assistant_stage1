# Neue Flask-App mit Funktionen aus dpa_modulA.py und UI wie in app_modulB.py
from flask import Flask, render_template, request, jsonify
import os, json, sys
from hdbcli import dbapi
from gen_ai_hub.proxy.langchain.openai import ChatOpenAI
from gen_ai_hub.proxy.langchain.init_models import init_embedding_model
from langchain.chains import RetrievalQA
from langchain.schema import Document
from langchain.text_splitter import CharacterTextSplitter
# Pfad hinzufügen, um dpa_modules zu importieren
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from dpa_modules.dpa_modulA import load_env_variables, load_pdf

app = Flask(__name__)
app.secret_key = os.urandom(24)
HISTORY_FILE = "input_history.json"
history = []
qa_chain = None
hana_database = None

def load_history():
    global history
    try:
        with open(HISTORY_FILE, "r", encoding="utf-8") as f:
            history = json.load(f)
    except FileNotFoundError:
        history = []

def save_history():
    with open(HISTORY_FILE, "w", encoding="utf-8") as f:
        json.dump(history, f, ensure_ascii=False, indent=2)

@app.route('/')
def index():
    return render_template('index.html', history=history)

@app.route('/process', methods=['POST'])
def process_input():
    global history, qa_chain
    user_input = request.form.get('input_text', '')
    if user_input and user_input not in history:
        history.append(user_input)
        save_history()
    if not qa_chain:
        return jsonify({"success": False, "message": "System nicht initialisiert."})
    try:
        answer = qa_chain.run(user_input)
        return jsonify({"success": True, "input": user_input, "output": answer})
    except Exception as e:
        return jsonify({"success": False, "message": str(e)})

@app.route('/history')
def get_history():
    return jsonify({"history": history})

@app.route('/initialize', methods=['POST'])
def initialize_system():
    global qa_chain, hana_database
    try:
        # Umgebungsvariablen laden
        config_file = os.path.expanduser("~/.aicore/config.json")
        load_env_variables(config_file)
        # LLM initialisieren
        llm = ChatOpenAI(proxy_model_name=os.getenv("AICORE_DEPLOYMENT_MODEL"))
        # Embedding-Modell initialisieren
        embeddings = init_embedding_model(os.getenv("AICORE_DEPLOYMENT_MODEL_EMBEDDING"))
        # HANA-Verbindungsparameter über os.environ beziehen
        try:
            hdb_host = os.environ["hdb_host_address"]
            hdb_user = os.environ["hdb_user"]
            hdb_password = os.environ["hdb_password"]
            hdb_port = int(os.environ["hdb_port"])
            table_name = os.environ["hdb_table_name"]
        except KeyError as e:
            return jsonify({"success": False, "message": f"Fehlende Umgebungsvariable: {e}"})
        # HANA-Verbindung aufbauen
        conn = dbapi.connect(
            address=hdb_host,
            port=hdb_port,
            user=hdb_user,
            password=hdb_password,
            autocommit=True
        )
        # VectorStore initialisieren
        from langchain_community.vectorstores.hanavector import HanaDB
        hana_database = HanaDB(
            embedding=embeddings,
            connection=conn,
            table_name=table_name
        )
        # PDF laden und in Chunks aufteilen
        docs = load_pdf("data/sample_accounting_guide.pdf")
        chunks = []
        # Split documents into chunks
        chunker = CharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        for doc in docs:
            for txt in chunker.split_text(doc.page_content):
                chunks.append(Document(page_content=txt, metadata=doc.metadata))
        # Index neu aufbauen
        hana_database.delete(filter={})
        hana_database.add_documents(chunks)
        # QA-Kette erstellen
        retriever = hana_database.as_retriever(search_kwargs={"k": 10})
        qa_chain = RetrievalQA.from_chain_type(
            llm=llm,
            retriever=retriever,
            chain_type="stuff",
            verbose=True
        )
        return jsonify({"success": True, "message": "Initialisierung erfolgreich"})
    except Exception as e:
        return jsonify({"success": False, "message": str(e)})

if __name__ == '__main__':
    load_history()
    app.run(debug=True, host='0.0.0.0', port=5000)
