# flake8: noqa
# BE_AI_DPA_APP.py
# Aggregated code cells from BE_AI_UC_DPA_modulB_PoC.ipynb

# B0.1 install py-packages
# (Entfernt Jupyter-Magics aus Python-Script; siehe requirements.txt für Abhängigkeiten)

# B0.2 load env-variables from config.json-file
import json
import os
from dotenv import load_dotenv  # hinzugefügt

def load_env_variables(config_file):
    """
    Lädt Umgebungsvariablen aus einer JSON-Konfigurationsdatei.

    Args:
        config_file (str): Pfad zur JSON-Datei mit Umgebungsvariablen.

    Returns:
        dict: Die geladenen Umgebungsvariablen als Schlüssel-Wert-Paare.

    Raises:
        FileNotFoundError: Wenn die Konfigurationsdatei nicht existiert.
        ValueError: Bei JSON-Dekodierungsfehlern.
    """
    if not os.path.exists(config_file):
        raise FileNotFoundError(f"The configuration file {config_file} does not exist.")
    try:
        with open(config_file, "r") as file:
            env_variables = json.load(file)
    except json.JSONDecodeError as e:
        raise ValueError(f"Error decoding JSON from {config_file}: {e}")
    for key, value in env_variables.items():
        if isinstance(value, dict):
            value = json.dumps(value)
        os.environ[key] = str(value)
    return env_variables

# B0.2 Test connection with env-Variables to SAP AI core
from gen_ai_hub.proxy.native.openai import embeddings as openai_embeddings
from .InputManager import InputManager
def test_ai_core_connection():
    """
    Testet die Verbindung zu SAP AI Core durch Senden einer Beispiel-Embedding-Anfrage.

    Returns:
        bool: True bei erfolgreicher Verbindung, False bei Fehler.
    """
    model_embedding_name = os.getenv("AICORE_DEPLOYMENT_MODEL_EMBEDDING")
    try:
        response = openai_embeddings.create(input="SAP Generative AI Hub is awesome!", model_name=model_embedding_name)
        print(response.data)
        return True
    except ValueError as e:
        print(f"Error: {e}")
        print("Ensure the model name matches an existing deployment in SAP AI Hub.")
        return False

# B0.3 Setup and test connection to HANA DB
from hdbcli import dbapi

def connect_to_hana_db():
    """
    Stellt eine Verbindung zur SAP HANA-Datenbank her.

    Returns:
        dbapi.Connection: Eine aktive HANA-Datenbankverbindung.

    Raises:
        ValueError: Wenn eine oder mehrere Verbindungsparameter fehlen.
    """
    hdb_host_address = str(os.getenv("hdb_host_address"))
    hdb_user = str(os.getenv("hdb_user"))
    hdb_password = str(os.getenv("hdb_password"))
    hdb_port = str(os.getenv("hdb_port"))
    print(f"hdb_host_address: {hdb_host_address}")
    print(f"hdb_user: {hdb_user}")
    print(f"hdb_port: {hdb_port}")
    if not all([hdb_host_address, hdb_user, hdb_password, hdb_port]):
        raise ValueError("One or more HANA DB connection parameters are missing.")
    assert hdb_port is not None, "hdb_port must not be None"
    hdb_port = int(hdb_port)
    hana_connection = dbapi.connect(address=hdb_host_address, port=hdb_port, user=hdb_user, password=hdb_password, autocommit=True)
    return hana_connection

# B0.4 Setup LLM-Connection to SAP AI-HUB
from gen_ai_hub.proxy.langchain.init_models import init_llm  # korrigierter Import
import re

def init_llm_connection():
    """
    Initialisiert eine Verbindung zum LLM-Modell basierend auf Umgebungsvariablen.

    Returns:
        LLM: Initialisiertes Sprachmodell.

    Raises:
        ValueError: Wenn der Modellname fehlt.
    """
    load_dotenv()
    aicore_model_name = str(os.getenv("AICORE_DEPLOYMENT_MODEL"))
    if not aicore_model_name:
        raise ValueError(f"LLM model name {aicore_model_name} missing.")
    llm = init_llm(model_name=aicore_model_name, max_tokens=4000, temperature=0)
    print(f"LLM loaded: {aicore_model_name}")
    return llm

# B0.4 Check Setup LLM-Connection
from langchain.prompts import PromptTemplate  # hinzugefügt
from langchain_core.output_parsers import StrOutputParser  # hinzugefügt

def test_llm_connection(llm):
    """
    Testet die Verbindung zum LLM mit einer Beispielanfrage.

    Args:
        llm (LLM): Das initialisierte LLM-Modell.

    Returns:
        Any: Antwort des Modells auf die Testanfrage.
    """
    template = """Question: {question}\nAnswer: Let's think step by step."""
    prompt = PromptTemplate(template=template, input_variables=["question"])
    chain = prompt | llm | StrOutputParser()
    response = chain.invoke({"question": "What is an ai-agent?"})
    print(response)
    return response

# B0.5 setup embedding-model
from gen_ai_hub.proxy.langchain.init_models import init_embedding_model

def init_embedding_model_connection():
    """
    Initialisiert das Embedding-Modell für Vektorraum-Repräsentationen.

    Returns:
        Embeddings: Objekt für Embedding-Erstellung.
    """
    ai_core_embedding_model_name = str(os.getenv("AICORE_DEPLOYMENT_MODEL_EMBEDDING"))
    embeddings = init_embedding_model(ai_core_embedding_model_name)
    print("Embedding model initialized: ", ai_core_embedding_model_name)
    return embeddings

# B0.6 create SAP HANA-VectorStore interface
from langchain_community.vectorstores.hanavector import HanaDB

def create_vector_store(embeddings, hana_connection):
    """
    Erstellt eine HANA-basierte Vektor-Datenbank (VectorStore).

    Args:
        embeddings: Embedding-Objekt zur Vektorgenerierung.
        hana_connection: Aktive HANA-Datenbankverbindung.

    Returns:
        HanaDB: Instanz des HANA-Vektor-Speichers.
    """
    vector_table_name = str(os.getenv("hdb_table_name"))
    hana_database = HanaDB(embedding=embeddings, connection=hana_connection, table_name=vector_table_name)
    print(f"VectorStore ready: {vector_table_name}")
    return hana_database

# B0.6 verify embeddings query
def verify_embeddings(hana_connection):
    """
    Überprüft gespeicherte Embeddings in der HANA-Datenbank.

    Args:
        hana_connection: Aktive HANA-Datenbankverbindung.

    Returns:
        list: Abgerufene Vektoren zur Überprüfung.
    """
    vector_table_name = str(os.getenv("hdb_table_name"))
    cursor = hana_connection.cursor()
    sql = f'SELECT VEC_TEXT, TO_NVARCHAR(VEC_VECTOR) FROM "{vector_table_name}" WHERE VEC_TEXT LIKE \'%Rückstellung%\''
    cursor.execute(sql)
    vectors = cursor.fetchall()
    print(vectors[5:10])
    return vectors

# # B0.7 setup class user interface
from .InputManager import InputManager

# B2 define Prompt (HTML and JSON versions)
from langchain.prompts import PromptTemplate

prompt_template_html = """

# Buchungssatz-Generator für Geschäftsfälle

## Aufgabe: Erstellung von Buchungssätzen für Geschäftsfälle anhand eines vorgegebenen Kontierungshandbuchs

Extrahiere anhand des vom Buchhalter angegebenen Geschäftsfalls die Kontierungsregeln für Konto-Soll und Konto Haben 
aus dem Kontierungshandbuch.

## Geschäftsfall:

Der Geschäftsfall wird vom Buchhalter wie folgt beschrieben:

{question}

## Kontierungsregeln:

Die Kontierungsregeln sind im Kontierungshandbuch wie folgt definiert:

{context}


## Wichtige Hinweise
Gehe für die Ermittlung des Buchungssatzes Schritt für Schritt vor und achte auf die folgenden Punkte:
- Identifiziere die Kategorie des Geschäftsfalls    
- Bestimme die wesentlichen finanziellen Merkmale und Beträge
- Wende die Kontierungsregel an
- Ermittle konkrete Konto-Nummern und -Bezeichnungen aus dem Kontext des Kontierungshandbuchs
- Bestimme die exakten Buchungsbeträge (inkl. Steuern falls zutreffend)
- Identifiziere eventuelle Besonderheiten oder Ausnahmen

## Vorgehen zur Lösung der Aufgabe

Beachte die folgenden Schritte bei der Lösung der Aufgabe

**Schritt 1 - Identifiziere die Geschäftsfall-Kategorie**

- Identifiziere Art der Transaktion und beteiligte Wirtschaftsgüter/Leistungen
- Bestimme alle relevanten Beträge inkl. Steuern
- Ermittle die Geschäftsfall-Kategorie

Beispiele für Geschäftsfall-Kategorien sind:

|Geschäftsfall-Kategorie|
|-----------------------|
|Warenverkauf auf Rechnung|
|Wareneinkauf auf Rechnung|
|Zahlung an Lieferanten|
|Zahlungseingang von Kunden|
|Lohn- und Gehaltszahlungen|
|Anschaffung von Anlagevermögen|
|Abschreibungen|
|Bildung von Rückstellungen|
|Rückstellung buchen in Schlussbilanz|
|Rückstellung buchen in Eröffnungsbilanz|
|Auflösung von Rückstellungen|
|Eingangsrechung buchen|
|Eingangsrechnung zahlen|
|Zahlung der Umsatzsteuer|

**Schritt 2 - Ermittle die Kontierungsregel anhand der Geschäftsfall-Kategorie aus dem Kontierungshandbuch.**

- Ermittle die Kontierungsregel mit Kontotyp-Soll und Kontotyp-Haben anhand der ermittelten Geschäftsfall-Kategorien
- Beachte die buchhalterischen Grundprinzipien:
    + Jede Buchung erfordert mindestens ein Soll- und Haben-Konto
    + Soll-Haben-Buchungslogik korrekt anwenden (Vermehrung Aktiva/Verminderung Passiva = SOLL; Verminderung Aktiva/Vermehrung Passiva = HABEN)
    + Buchhalterische Vollständigkeit sicherstellen (Summe SOLL = Summe HABEN)
    + Buchungssätze nach Reihenfolge der Geschäftsfälle ordnen

Beispiele von Kontierungsregeln für Geschäftsfall-Kategoreien sind:

|Geschäftsfall-Kategorie|Kontotyp-Soll|Kontotyp-Haben|
|-----------------------|----------|-----------|
|Warenverkauf auf Rechnung|Forderungen aus Lieferungen und Leistungen|Umsatzerlöse und Umsatzsteuer|
|Wareneinkauf auf Rechnung|Wareneinsatz/Materialaufwand und Vorsteuer|Verbindlichkeiten aus Lieferungen und Leistungen|
|Zahlung an Lieferanten|Verbindlichkeiten aus Lieferungen und Leistungen|Bank oder Kasse|
|Zahlungseingang von Kunden|Bank oder Kasse|Forderungen aus Lieferungen und Leistungen|
|Lohn- und Gehaltszahlungen|Personalaufwand|Bank und diverse Verbindlichkeiten (Lohnsteuer, Sozialversicherung)|
|Anschaffung von Anlagevermögen|Anlagevermögen und Vorsteuer|Bank oder Verbindlichkeiten|
|Abschreibungen|Abschreibungsaufwand|Kumulierte Abschreibungen (Wertberichtigung Anlagevermögen)|
|Bildung von Rückstellungen|Aufwand für Rückstellungen|Rückstellungen|
|Auflösung von Rückstellungen|Rückstellungen|Sonstige betriebliche Erträge oder Aufwandskonto|
|Rückstellung buchen in Schlussbilanz (Bilanz)|Rückstellungen|Schlussbilanz-Konto|
|Rückstellung buchen in Schlussbilanz (GuV)|GuV-Konto |Aufwand für Lieferung und Leistung|
|Rückstellung buchen in Eröffnungsbilanz|Eröffnungsbilanz-Konto|Aufwand für Lieferung und Leistung|
|Eingangsrechung buchen|Aufwand für Lieferung und Leistung|Verbindlichkeiten oder Kreditor|
|Eingangsrechnung zahlen |Verbindlichkeiten oder Kreditor|Bank|
|Zahlung der Umsatzsteuer|Umsatzsteuer|Bank|


**Schritt 3 - Extrahiere die relevante Kontierung für Kontotyp-Soll und Kontotyp-Haben der Kontierungsregel aus dem Kontierungshandbuch**
- Ermittle die konkreten und exakten Konto-Nummern und Konto-Bezeichnungen für Kontotyp-Soll und Kontotyp-Haben aus dem Kontierungshandbuch anhand
  der gefundenen Kontierungsregel 
- Bestimme die exakten Buchungsbeträge aus den Informationen des Geschäftsfalls
- Beachte Spezialregelungen (z.B. für Steuern, Rückstellungen, Abschreibungen)

**Schritt 4 - Prüfe die Qualität des Ergebnisses**
- Prüfe die Kontierung (Konto Soll an Konto Haben)
- Prüfe die doppelte Buchführung (Betrag Soll = Betrag Haben)
- Stelle Übereinstimmung mit gesetzlichen Anforderungen sicher
- Verifiziere die inhaltliche Korrektheit der Kontierung

**Schritt 5 - Gib die Ergebnisse aus**

- Entferne Duplikate und redundante Informationen
- Priorisiere die relevantesten und spezifischsten Kontierungen
- Strukturiere das Ergebnis klar und übersichtlich

- Gib folgende Informationen des Ergebnisses aus:
    + Geschäftsfallbezeichnung
    + Geschäftsfall-Kategorie
    + Exakte Kontierungsinformation mit Kontonummern für Konto-Soll und Konto-Haben und den Konten-Bezeichnungen
    + Soll-Haben-Beziehung mit Beträgen

- Prüfe die Ausgabe entsprechend der oben ausgeführten Schritte und achte darauf, dass: 
    die Informationen klar und strukturiert präsentiert werden,
    + die Ausgabe in HTML-Format erfolgt und keine Code-Block-Markierungen enthält,
    + die Kontierungsinformationen vollständig und korrekt sind,
    + die Ausgabe keine überflüssigen Informationen enthält,
    + die Ausgabe keine persönlichen Daten oder sensiblen Informationen enthält,
    + die Ausgabe keine nicht relevanten Informationen enthält.

- Antwortformat:
<div class="buchungssatz">
  <h2>Geschäftsfall: [PRÄZISE BEZEICHNUNG GESCHÄFTSFALL]</h2>
  <h3>Geschäftsfall-Kategorie: [GESCHÄFTSFALL-KATEGORIE]</h3>
  <div class="kontierung">
    <table>
      <tr>
        <th>Soll</th>
        <th>Haben</th>
        <th>Betrag</th>
      </tr>
      <tr>
        <td>[KONTO-NR] - [BEZEICHNUNG]</td>
        <td>[KONTO-NR] - [BEZEICHNUNG]</td>
        <td>[BETRAG] [WÄHRUNG]</td>
      </tr>
      <!-- Weitere Zeilen bei Bedarf -->
    </table>
  </div>
  <div class="erläuterung">
    <p>[KURZE BEGRÜNDUNG DER KONTIERUNG]</p>
  </div>
</div>

## Wenn keine passende Kontierung gefunden werden kann:
<div class="keine-kontierung">
  <p>Für diesen Geschäftsfall konnte keine passende Kontierung in den bereitgestellten Regeln ermittelt werden. 
  Es fehlen folgende Informationen: [FEHLENDE INFORMATIONEN]</p>
</div>
"""
prompt_template_html = PromptTemplate(template=prompt_template_html, input_variables=["context","question"])
print("Prompt HTML set")

prompt_template_json = """

# Buchungssatz-Generator für Geschäftsfälle

## Aufgabe: Erstellung von Buchungssätzen für Geschäftsfälle anhand eines vorgegebenen Kontierungshandbuchs

Extrahiere anhand des vom Buchhalter angegebenen Geschäftsfalls die Kontierungsregeln für Konto-Soll und Konto Haben 
aus dem Kontierungshandbuch.

## Geschäftsfall:

Der Geschäftsfall wird vom Buchhalter wie folgt beschrieben:

{question}

## Kontierungsregeln:

Die Kontierungsregeln sind im Kontierungshandbuch wie folgt definiert:

{context}


## Wichtige Hinweise
Gehe für die Ermittlung des Buchungssatzes Schritt für Schritt vor und achte auf die folgenden Punkte:
- Identifiziere die Kategorie des Geschäftsfalls    
- Bestimme die wesentlichen finanziellen Merkmale und Beträge
- Wende die Kontierungsregel an
- Ermittle konkrete Konto-Nummern und -Bezeichnungen aus dem Kontext des Kontierungshandbuchs
- Bestimme die exakten Buchungsbeträge (inkl. Steuern falls zutreffend)
- Identifiziere eventuelle Besonderheiten oder Ausnahmen

## Vorgehen zur Lösung der Aufgabe

Beachte die folgenden Schritte bei der Lösung der Aufgabe

**Schritt 1 - Identifiziere die Geschäftsfall-Kategorie**

- Identifiziere Art der Transaktion und beteiligte Wirtschaftsgüter/Leistungen
- Bestimme alle relevanten Beträge inkl. Steuern
- Ermittle die Geschäftsfall-Kategorie

Beispiele für Geschäftsfall-Kategorien sind:

|Geschäftsfall-Kategorie|
|-----------------------|
|Warenverkauf auf Rechnung|
|Wareneinkauf auf Rechnung|
|Zahlung an Lieferanten|
|Zahlungseingang von Kunden|
|Lohn- und Gehaltszahlungen|
|Anschaffung von Anlagevermögen|
|Abschreibungen|
|Bildung von Rückstellungen|
|Rückstellung buchen in Schlussbilanz|
|Rückstellung buchen in Eröffnungsbilanz|
|Auflösung von Rückstellungen|
|Eingangsrechung buchen|
|Eingangsrechnung zahlen|
|Zahlung der Umsatzsteuer|

**Schritt 2 - Ermittle die Kontierungsregel anhand der Geschäftsfall-Kategorie aus dem Kontierungshandbuch.**

- Ermittle die Kontierungsregel mit Kontotyp-Soll und Kontotyp-Haben anhand der ermittelten Geschäftsfall-Kategorien
- Beachte die buchhalterischen Grundprinzipien:
    + Jede Buchung erfordert mindestens ein Soll- und Haben-Konto
    + Soll-Haben-Buchungslogik korrekt anwenden (Vermehrung Aktiva/Verminderung Passiva = SOLL; Verminderung Aktiva/Vermehrung Passiva = HABEN)
    + Buchhalterische Vollständigkeit sicherstellen (Summe SOLL = Summe HABEN)
    + Buchungssätze nach Reihenfolge der Geschäftsfälle ordnen

Beispiele von Kontierungsregeln für Geschäftsfall-Kategoreien sind:

|Geschäftsfall-Kategorie|Kontotyp-Soll|Kontotyp-Haben|
|-----------------------|----------|-----------|
|Warenverkauf auf Rechnung|Forderungen aus Lieferungen und Leistungen|Umsatzerlöse und Umsatzsteuer|
|Wareneinkauf auf Rechnung|Wareneinsatz/Materialaufwand und Vorsteuer|Verbindlichkeiten aus Lieferungen und Leistungen|
|Zahlung an Lieferanten|Verbindlichkeiten aus Lieferungen und Leistungen|Bank oder Kasse|
|Zahlungseingang von Kunden|Bank oder Kasse|Forderungen aus Lieferungen und Leistungen|
|Lohn- und Gehaltszahlungen|Personalaufwand|Bank und diverse Verbindlichkeiten (Lohnsteuer, Sozialversicherung)|
|Anschaffung von Anlagevermögen|Anlagevermögen und Vorsteuer|Bank oder Verbindlichkeiten|
|Abschreibungen|Abschreibungsaufwand|Kumulierte Abschreibungen (Wertberichtigung Anlagevermögen)|
|Bildung von Rückstellungen|Aufwand für Rückstellungen|Rückstellungen|
|Auflösung von Rückstellungen|Rückstellungen|Sonstige betriebliche Erträge oder Aufwandskonto|
|Rückstellung buchen in Schlussbilanz (Bilanz)|Rückstellungen|Schlussbilanz-Konto|
|Rückstellung buchen in Schlussbilanz (GuV)|GuV-Konto |Aufwand für Lieferung und Leistung|
|Rückstellung buchen in Eröffnungsbilanz|Eröffnungsbilanz-Konto|Aufwand für Lieferung und Leistung|
|Eingangsrechung buchen|Aufwand für Lieferung und Leistung|Verbindlichkeiten oder Kreditor|
|Eingangsrechnung zahlen |Verbindlichkeiten oder Kreditor|Bank|
|Zahlung der Umsatzsteuer|Umsatzsteuer|Bank|


**Schritt 3 - Extrahiere die relevante Kontierung für Kontotyp-Soll und Kontotyp-Haben der Kontierungsregel aus dem Kontierungshandbuch**
- Ermittle die konkreten und exakten Konto-Nummern und Konto-Bezeichnungen für Kontotyp-Soll und Kontotyp-Haben aus dem Kontierungshandbuch anhand
  der gefundenen Kontierungsregel 
- Bestimme die exakten Buchungsbeträge aus den Informationen des Geschäftsfalls
- Beachte Spezialregelungen (z.B. für Steuern, Rückstellungen, Abschreibungen)

**Schritt 4 - Prüfe die Qualität des Ergebnisses**
- Prüfe die Kontierung (Konto Soll an Konto Haben)
- Prüfe die doppelte Buchführung (Betrag Soll = Betrag Haben)
- Stelle Übereinstimmung mit gesetzlichen Anforderungen sicher
- Verifiziere die inhaltliche Korrektheit der Kontierung

**Schritt 5 - Gib die Ergebnisse aus**

- Entferne Duplikate und redundante Informationen
- Priorisiere die relevantesten und spezifischsten Kontierungen
- Strukturiere das Ergebnis klar und übersichtlich

- Gib folgende Informationen des Ergebnisses aus:
    + Geschäftsfallbezeichnung
    + Geschäftsfall-Kategorie
    + Exakte Kontierungsinformation mit Kontonummern für Konto-Soll und Konto-Haben und den Konten-Bezeichnungen
    + Soll-Haben-Beziehung mit Beträgen

- Prüfe die Ausgabe entsprechend der oben ausgeführten Schritte und achte darauf, dass: 
    die Informationen klar und strukturiert präsentiert werden,
    + die Ausgabe nur absolut valides JSON-Format ohne Markdown-Formatierung oder Codeblöcke enthält
    + nur das JSON zurückgegeben wird.
    + die Kontierungsinformationen vollständig und korrekt sind,
    + die Ausgabe keine überflüssigen Informationen enthält,
    + die Ausgabe keine persönlichen Daten oder sensiblen Informationen enthält,
    + die Ausgabe keine nicht relevanten Informationen enthält.

## Antwortformat (JSON):

Du sollst ein JSON im folgenden Format erzeugen.
Für erfolgreiche Kontierung:
{{
  "geschaeftsfall": {{
    "bezeichnung": "PRÄZISE BEZEICHNUNG",
    "buchungen": [
      {{
        "soll": {{
          "kontonummer": "KONTO-NR",
          "bezeichnung": "BEZEICHNUNG"
        }},
        "haben": {{
          "kontonummer": "KONTO-NR",
          "bezeichnung": "BEZEICHNUNG"
        }},
        "betrag": "BETRAG",
        "waehrung": "WÄHRUNG"
      }}
    ],
    "erlaeuterung": "KURZE BEGRÜNDUNG DER KONTIERUNG"
  }}
}}

## Wenn keine passende Kontierung gefunden werden kann:
{{
  "fehler": {{
    "meldung": "Keine passende Kontierung gefunden",
    "fehlende_informationen": ["INFORMATION_1", "INFORMATION_2"]
  }}
}}

"""
prompt_template_json = PromptTemplate(template=prompt_template_json, input_variables=["context","question"])
print("Prompt JSON set")

# B3 answer: RetrievalQA
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)
from langchain.chains import RetrievalQA

def create_qa_chain(llm, hana_database, prompt_template, count_retrieved_documents=10):
    """
    Erstellt eine RetrievalQA-Chain für Frage-Antwort-Anwendungen.

    Args:
        llm: Das Sprachmodell für die QA-Kette.
        hana_database: Vektor-Datenbank zur Kontextabfrage.
        prompt_template: Vorlage für die Eingabeaufforderung.
        count_retrieved_documents (int): Anzahl abzurufender Dokumente.

    Returns:
        RetrievalQA: Konfigurierte QA-Kette.
    """
    chain_type_kwargs = {"prompt": prompt_template}
    retriever = hana_database.as_retriever(search_kwargs={"k": count_retrieved_documents})
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm, 
        retriever=retriever, 
        chain_type="stuff", 
        chain_type_kwargs=chain_type_kwargs, 
        verbose=True
    )
    return qa_chain

# Wenn diese Datei direkt ausgeführt wird, starte die Jupyter-basierte UI
if __name__ == "__main__":
    config_file = "/home/user/.aicore/config.json"
    try:
        env_variables = load_env_variables(config_file)
        print(f"Loaded environment variables: {env_variables}")
    except (FileNotFoundError, ValueError) as e:
        print(e)
    
    test_ai_core_connection()
    
    hana_connection = connect_to_hana_db()
    
    llm = init_llm_connection()
    test_llm_connection(llm)
    
    embeddings = init_embedding_model_connection()
    hana_database = create_vector_store(embeddings, hana_connection)
    verify_embeddings(hana_connection)
    
    # Instanziiere InputManager, um die Fehler zu beheben
    input_manager = InputManager()

    prompt_template = prompt_template_html
    
    # B1 display UI
    input_manager.display_widget()
    
    # B3.1 run LLM with prompt template
    count_retrieved_documents = 10
    question = input_manager.get_current_input()
    qa_chain = create_qa_chain(llm, hana_database, prompt_template, count_retrieved_documents)
    answer = qa_chain.run(question)
    print(answer)
    
    # B4 output: display answer
    input_manager.update_output(answer)
    input_manager.display_widget()
