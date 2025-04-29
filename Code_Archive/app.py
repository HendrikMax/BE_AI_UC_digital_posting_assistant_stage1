import os
import json
from typing import List, Dict
from hdbcli import dbapi
from gen_ai_hub.proxy.langchain.openai import ChatOpenAI
from gen_ai_hub.proxy.native.openai import embeddings

class AccountingAssistant:
    def __init__(self):
        self._load_config()
        self._setup_connections()
        
    def _load_config(self):
        """Lädt die Konfiguration aus der config.json Datei"""
        config_file = "/home/user/.aicore/config.json"
        try:
            with open(config_file, 'r') as file:
                self.config = json.load(file)
                for key, value in self.config.items():
                    os.environ[key] = str(value)
        except Exception as e:
            raise ValueError(f"Fehler beim Laden der Konfiguration: {str(e)}")

    def _setup_connections(self):
        """Richtet die Verbindungen zu HANA DB und LLM ein"""
        # HANA DB Verbindung
        self.hana_connection = dbapi.connect(
            address=os.getenv("hdb_host_address"),
            port=int(os.getenv("hdb_port")),
            user=os.getenv("hdb_user"),
            password=os.getenv("hdb_password"),
            autocommit=True,
            sslValidateCertificate=False
        )
        
        # LLM Verbindung
        self.llm = ChatOpenAI(proxy_model_name=os.getenv("AICORE_DEPLOYMENT_MODEL"))

    def get_booking_information(self, business_case: str) -> Dict:
        """
        B1: Verarbeitet die Buchungsinformationen aus dem Geschäftsfall
        
        Args:
            business_case: Der Geschäftsfall als Text
            
        Returns:
            Dict mit den extrahierten Buchungsinformationen
        """
        # Hier wird der Geschäftsfall analysiert und strukturiert
        # Dies ist ein Platzhalter für die tatsächliche Implementierung
        return {
            "business_case": business_case,
            "extracted_info": "Extrahiert aus dem Geschäftsfall"
        }

    def retrieve_accounting_assignments(self, booking_info: Dict) -> List[Dict]:
        """
        B2: Ruft relevante Buchungszuordnungen aus der HANA DB ab
        
        Args:
            booking_info: Die extrahierten Buchungsinformationen
            
        Returns:
            Liste von relevanten Buchungszuordnungen
        """
        try:
            cursor = self.hana_connection.cursor()
            query = f"""
                SELECT * FROM {os.getenv('hdb_table_name')}
                WHERE business_case LIKE '%{booking_info['business_case']}%'
            """
            cursor.execute(query)
            results = cursor.fetchall()
            return [dict(zip([column[0] for column in cursor.description], row)) 
                   for row in results]
        except Exception as e:
            raise Exception(f"Fehler beim Abrufen der Buchungszuordnungen: {str(e)}")

    def create_accounting_assignments(self, booking_info: Dict, 
                                    relevant_assignments: List[Dict]) -> str:
        """
        B3: Erstellt Buchungszuordnungen mit Hilfe des LLM
        
        Args:
            booking_info: Die extrahierten Buchungsinformationen
            relevant_assignments: Die relevanten Buchungszuordnungen
            
        Returns:
            String mit den generierten Buchungszuordnungen
        """
        prompt = f"""
        Basierend auf dem folgenden Geschäftsfall und den relevanten Buchungszuordnungen,
        erstelle eine passende Buchungszuordnung:
        
        Geschäftsfall: {booking_info['business_case']}
        
        Relevante Zuordnungen: {relevant_assignments}
        
        Bitte erstelle eine detaillierte Buchungszuordnung mit:
        - Konten
        - Beträgen
        - Buchungstexten
        """
        
        response = self.llm.predict(prompt)
        return response

    def process_business_case(self, business_case: str) -> str:
        """
        B4: Verarbeitet den gesamten Geschäftsfall und gibt die Antwort zurück
        
        Args:
            business_case: Der Geschäftsfall als Text
            
        Returns:
            String mit der finalen Antwort
        """
        # B1: Buchungsinformationen extrahieren
        booking_info = self.get_booking_information(business_case)
        
        # B2: Relevante Zuordnungen abrufen
        relevant_assignments = self.retrieve_accounting_assignments(booking_info)
        
        # B3: Neue Zuordnungen erstellen
        accounting_assignments = self.create_accounting_assignments(
            booking_info, relevant_assignments)
        
        return accounting_assignments

def main():
    # Beispiel für die Verwendung
    assistant = AccountingAssistant()
    
    business_case = """
    Beispiel-Geschäftsfall:
    Wir haben am 15.03.2024 Büromaterial im Wert von 500€ gekauft.
    Die Rechnung wurde sofort bezahlt.
    """
    
    result = assistant.process_business_case(business_case)
    print("Ergebnis der Buchungszuordnung:")
    print(result)

if __name__ == "__main__":
    main() 