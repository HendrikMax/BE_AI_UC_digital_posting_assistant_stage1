#
#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Flask-Implementierung des Digitalen Buchungsassistenten - Modul B App für Ermittlung der Kontierung aus der Kontierungsrichtlinie
# für die SAP S/4HANA Cloud
# Autor: [Hendrik Max]
# Datum: [30.04.2025]
# Beschreibung: 
# Diese Anwendung ermöglicht es Benutzern, Anahnd einer GEschäftsfall-Beschreibung eine Kontierung
# zu ermitteln. Die Antworten werden mithilfe eines Retrievals auf eine Vektordatenbank generiert. 
# Die Anwendung speichert auch die Eingabehistorie der Benutzer in einer JSON-Datei, 
# um eine bessere Benutzererfahrung zu bieten.
# 
# Architektur:
# Die RAG-Anwendung verwendet ein Large Language Model (LLM) und ein Embedding-Modell,
# und eine HANA-Datenbank als Vektor-Datenbank. 

# Technologien:
# - Flask: Ein leichtgewichtiges Web-Framework für Python, das die Erstellung von Webanwendungen erleichtert.
# - Langchain: Eine Bibliothek, die den Zugriff auf verschiedene Large Language Models (LLMs) und Embedding-Modelle ermöglicht.
# - SAP HANA: Eine In-Memory-Datenbank, die als Vektordatenbank für die Speicherung und den Abruf von Daten verwendet wird.
# - JSON: Ein Standardformat für den Austausch von Daten, das in dieser Anwendung verwendet wird, um die Eingabehistorie zu speichern.
# - HTML/CSS: Die Anwendung verwendet HTML und CSS für die Benutzeroberfläche.
# - Werkzeug: Eine Sammlung von WSGI-Hilfsfunktionen, die in Flask verwendet werden.
#
# Aufruf:
# Importiere die benötigten Module und Standardbibliotheken:
# pip install -r requirements.txt
# Starte App  Modul B
# cd /home/user/projects/BE_AI_UC_digital_posting_assistant_build1/BE_AI_DPA_APP && python3 app_modulB.py


from flask import Flask, render_template, request, jsonify, session
import json
import os
import sys
from datetime import datetime

# NEU: RetrievalQA importieren
from langchain.chains import RetrievalQA

# Path zum Hauptverzeichnis hinzufügen, um Modulimporte zu ermöglichen
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
# NEU: Importiere die benötigten Module
from dpa_modules.dpa_modulB import (
    load_env_variables, 
    init_llm, 
    init_embedding_model, 
    HanaDB, 
    prompt_template_html,
    create_qa_chain
)

# Initialisiere Flask
app = Flask(__name__)
app.secret_key = os.urandom(24)

# Speicherort für die Eingabehistorie
HISTORY_FILE = "input_history_modulB.json"

# Globale Variablen für die Anwendung
input_text = ""
history = []
qa_chain = None
llm = None
hana_database = None

# Lade die Eingabehistorie
def load_history():
    global history
    try:
        with open(HISTORY_FILE, "r", encoding="utf-8") as f:
            history = json.load(f)
    except FileNotFoundError:
        history = []
        print("Keine gespeicherte Historie gefunden.")

# Speichere die Eingabehistorie
def save_history():
    with open(HISTORY_FILE, "w", encoding="utf-8") as f:
        json.dump(history, f, ensure_ascii=False, indent=2)

# Hauptroute - Startseite
@app.route('/')
def index():
    return render_template('index_modulB.html', history=history)

# Route für die Verarbeitung der Eingabe
@app.route('/process', methods=['POST'])
def process_input():
    global input_text, history, qa_chain
    
    # Hole die Eingabe aus dem Formular
    input_text = request.form.get('input_text', '')
    
    # Füge die Eingabe zur Historie hinzu
    if input_text and input_text not in history:
        history.append(input_text)
        save_history()
    
    # Wenn qa_chain nicht initialisiert wurde, Fehlermeldung zurückgeben
    if not qa_chain:
        return jsonify({
            "success": False,
            "message": "Das System wurde noch nicht initialisiert. Bitte starten Sie die Anwendung neu."
        })
    
    try:
        # Führe die Anfrage durch
        answer = qa_chain.run(input_text)
        return jsonify({
            "success": True,
            "input": input_text,
            "output": answer
        })
    except Exception as e:
        return jsonify({
            "success": False,
            "message": f"Fehler bei der Verarbeitung: {str(e)}"
        })

# Route für den Abruf der Historie
@app.route('/history')
def get_history():
    return jsonify({"history": history})

# Route zum Initialisieren des Systems
@app.route('/initialize', methods=['POST'])
def initialize_system():
    global qa_chain, llm, hana_database
    
    # Hier würde die Initialisierungslogik aus BE_AI_DPA_APP_v1.py stehen
    # In einer echten Implementierung würde dies möglicherweise async passieren
    
    try:
        # Umgebungsvariablen laden
        config_file = "/home/user/.aicore/config.json"
        env_variables = load_env_variables(config_file)
        
        # LLM und Embedding-Modelle initialisieren
        aicore_model_name = str(os.getenv("AICORE_DEPLOYMENT_MODEL"))
        llm = init_llm(model_name=aicore_model_name, max_tokens=4000, temperature=0)
        
        # HANA-DB und Vector Store initialisieren
        ai_core_embedding_model_name = str(os.getenv("AICORE_DEPLOYMENT_MODEL_EMBEDDING"))
        embeddings = init_embedding_model(ai_core_embedding_model_name)
        
        # Verbindung zur HANA-DB herstellen
        from hdbcli import dbapi
        hdb_host_address = str(os.getenv("hdb_host_address"))
        hdb_user = str(os.getenv("hdb_user"))
        hdb_password = str(os.getenv("hdb_password"))
        hdb_port = int(os.getenv("hdb_port"))
        hana_connection = dbapi.connect(address=hdb_host_address, port=hdb_port, user=hdb_user, password=hdb_password, autocommit=True)
        
        # Vector Store initialisieren
        vector_table_name = str(os.getenv("hdb_table_name"))
        hana_database = HanaDB(embedding=embeddings, connection=hana_connection, table_name=vector_table_name)
        
        # RetrievalQA Chain erstellen
        count_retrieved_documents = 10
        chain_type_kwargs = {"prompt": prompt_template_html}
        retriever = hana_database.as_retriever(search_kwargs={"k": count_retrieved_documents})
        qa_chain = RetrievalQA.from_chain_type(llm=llm, retriever=retriever, chain_type="stuff", chain_type_kwargs=chain_type_kwargs, verbose=True)
        
        return jsonify({
            "success": True,
            "message": "System erfolgreich initialisiert"
        })
    except Exception as e:
        return jsonify({
            "success": False,
            "message": f"Fehler bei der Initialisierung: {str(e)}"
        })

# Starte die Anwendung, wenn das Skript direkt ausgeführt wird
if __name__ == '__main__':
    # Lade die Eingabehistorie beim Start
    load_history()
    
    # Starte den Flask-Server im Debug-Modus
    app.run(debug=True, host='0.0.0.0', port=5000)
