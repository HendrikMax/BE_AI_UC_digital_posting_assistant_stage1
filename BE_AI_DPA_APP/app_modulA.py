#
#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Flask-Implementierung des Digitalen Buchungsassistenten - Modul A App für Vektoriesierung der Kontierungsrichtlinie
# in einer SAP S/4HANA Cloud-Datenbank
# Autor: [Hendrik Max]
# Datum: [30.04.2025]
# Beschreibung: 
# Diese Anwendung ermöglicht es Benutzern, eine PDF-Datei hochzuladen, die dann in Chunks zerlegt und in einer SAP HANA-Datenbank gespeichert wird.
# Die Anwendung speichert auch die Eingabehistorie der Benutzer in einer JSON-Datei,
# um eine bessere Benutzererfahrung zu bieten.
#
# Architektur:
# RAG-Anwendung, die ein Embedding-Modell und eine HANA-Datenbank als Vektordatenbank verwendet.
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
# pip install -r /home/user/projects/BE_AI_UC_digital_posting_assistant_build1/BE_AI_DPA_APP/requirements.txt
# Starte App  Modul A
# cd /home/user/projects/BE_AI_UC_digital_posting_assistant_build1/BE_AI_DPA_APP && python3 app_modulA.py
#

# Importiere die benötigten Module
from flask import Flask, render_template, request, jsonify
from numpy import save
from werkzeug.utils import secure_filename
import os, json, sys

# Pfad hinzufügen, um dpa_modules zu importieren
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from dpa_modules.dpa_modulA import setup_hana_connection, setup_llm, setup_embedding_model, setup_hana_vectorstore
from dpa_modules.dpa_modulA import load_env_variables, load_pdf, semantic_chunking, reload_embeddings


# Flask-App initialisieren
# Flask-Server für Modul A
app = Flask(__name__)
app.secret_key = os.urandom(24)
HISTORY_FILE_MODULA = "input_history_modulA.json"
history_modula = []
hana_database = None

# Speicherort für die hochgeladenen Dateien
# Sicherstellen, dass der Upload-Ordner existiert   
UPLOAD_FOLDER = os.path.join(os.path.dirname(__file__), 'static', 'uploads')
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
ALLOWED_EXTENSIONS = {'pdf'}

# Hilfsfunktion zum Prüfen erlaubter Dateitypen
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# Hilfsfunktion zum Laden der Hisorie-Datei
def load_history_modula():
    global history_modula
    try:
        with open(HISTORY_FILE_MODULA, "r", encoding="utf-8") as f:
            history_modula = json.load(f)
    except FileNotFoundError:
        history_modula = []
        # Datei anlegen, wenn sie nicht existiert
        with open(HISTORY_FILE_MODULA, "w", encoding="utf-8") as f:
            json.dump(history_modula, f, ensure_ascii=False, indent=2)

# Hilfsfunktion zum Speichern der History-Datei
def save_history_modula():
    with open(HISTORY_FILE_MODULA, "w", encoding="utf-8") as f:
        json.dump(history_modula, f, ensure_ascii=False, indent=2)

# Route: Initialisierung System (Laden Umgebungsvariablen, Setup LLM, Embedding, HANA-VectorStore)
@app.route('/initialize', methods=['POST'])
def initialize_system():
    global llm, embeddings, hana_connection, hana_database
    try:
        config_file = os.path.expanduser("~/.aicore/config.json")
        load_env_variables(config_file)
        llm = setup_llm()
        embeddings = setup_embedding_model()
        hana_connection = setup_hana_connection()
        hana_database = setup_hana_vectorstore(embeddings, hana_connection)        
        return jsonify({"success": True, "message": "Initialisierung erfolgreich. Bitte PDF hochladen."})
    except Exception as e:
        return jsonify({"success": False, "message": str(e)})

# Route: Upload Datei (PDF)
@app.route('/upload_pdf', methods=['POST'])
def upload_pdf():
    global docs, filename, filepath
    if 'file' not in request.files:
        return jsonify({"success": False, "message": "Keine Datei hochgeladen."})
    file = request.files['file']
    if file.filename == '':
        return jsonify({"success": False, "message": "Keine Datei ausgewählt."})
    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        # PDF laden
        docs = load_pdf(filepath)
        # Aktuelle Dateiliste nach Upload holen
        files = os.listdir(app.config['UPLOAD_FOLDER'])
        return jsonify({
            "success": True,
            "message": "PDF-Datei erfolgreich geladen.",
            "filename": filename,
            "files": files
        })
    else:
        return jsonify({"success": False, "message": "Ungültiger Dateityp."})
    
# Index-Route: Zeigt Upload von Datei an
@app.route('/')
def index():
    # Zeige aktuelle Dateiliste im Upload-Ordner an
    files = os.listdir(app.config['UPLOAD_FOLDER']) if os.path.exists(app.config['UPLOAD_FOLDER']) else []
    # Zeige den zuletzt hochgeladenen Dateinamen an, falls vorhanden
    last_filename = files[-1] if files else None
    return render_template('index.html', files=files, last_filename=last_filename)


# Route: Verarbeitung (Chunking & Upload in HANA-DB)
@app.route('/process_file', methods=['POST'])
def process_file():
    global hana_database, docs, embeddings, llm, filename, filepath
    # filename = request.json.get('filename')
    if not filename:
        return jsonify({"success": False, "message": "Keine Datei angegeben."})
    # filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    if not os.path.exists(filepath):
        return jsonify({"success": False, "message": "Datei nicht gefunden."})
    if not hana_database:
        return jsonify({"success": False, "message": "System nicht initialisiert."})
    try:
        # Chuncks erstellen
        text_chunks = semantic_chunking(docs, embeddings)
        anzahl_chunks = len(text_chunks)
        # Chunks in HANA-DB hochladen
        reload_embeddings(hana_database, text_chunks)
        # --- History aktualisieren ---
        global history_modula
        if filename not in history_modula:
            history_modula.append(filename)
            save_history_modula()
        # ---
        return jsonify({
            "success": True,
            "message": f"Verarbeitung abgeschlossen. {anzahl_chunks} Chunks wurden in HANA-Datenbank hochgeladen.",
            "anzahl_chunks": anzahl_chunks
        })
    except Exception as e:
        return jsonify({"success": False, "message": str(e)})

# Route: History laden (inkl. Dateiliste)
@app.route('/history_modula', methods=['GET'])
def get_history_modula():
    load_history_modula()
    return render_template('index_modulA.html', history_modula=history_modula if history_modula is not None else [])

# Route: History speichern (optional, falls benötigt)
@app.route('/save_history_modula', methods=['POST'])
def save_history_modula_route():
    global history_modula
    data = request.get_json()
    if data and 'history_modula' in data:
        history_modula = data['history_modula']
        save_history_modula()
        return jsonify({"success": True})
    return jsonify({"success": False, "message": "Keine History-Daten erhalten."})

# Route: History laden (für AJAX-Anfrage)
@app.route('/load_history_modula', methods=['GET'])
def load_history_modula_route():
    load_history_modula()
    return jsonify({"success": True, "history_modula": history_modula if history_modula is not None else []})

# Verarbeitung Hauptanwendung
if __name__ == '__main__':
    # Stelle sicher, dass die History für Modul A beim Start geladen wird
    load_history_modula()
    app.run(debug=True, host='0.0.0.0', port=5000)
    # Speichere die History für Modul A beim Beenden der App
    save_history_modula()

