import tkinter as tk
from tkinter import ttk, font
import json
import os
from typing import List, Dict
from hdbcli import dbapi
from gen_ai_hub.proxy.langchain.openai import ChatOpenAI
from gen_ai_hub.proxy.native.openai import embeddings

class MaterialStyle:
    """Google Material Design Farbpalette und Stile"""
    PRIMARY_COLOR = "#1976D2"  # Blau 700
    SECONDARY_COLOR = "#BBDEFB"  # Blau 100
    BACKGROUND_COLOR = "#FFFFFF"
    TEXT_COLOR = "#212121"
    BUTTON_TEXT_COLOR = "#FFFFFF"
    PADDING = 20
    BUTTON_PADDING = 10
    
    @staticmethod
    def configure_styles():
        style = ttk.Style()
        style.configure("Material.TFrame", background=MaterialStyle.BACKGROUND_COLOR)
        style.configure("Material.TLabel", 
                       background=MaterialStyle.BACKGROUND_COLOR,
                       foreground=MaterialStyle.TEXT_COLOR,
                       padding=MaterialStyle.PADDING)
        style.configure("Material.TButton",
                       background=MaterialStyle.PRIMARY_COLOR,
                       foreground=MaterialStyle.BUTTON_TEXT_COLOR,
                       padding=MaterialStyle.BUTTON_PADDING)
        style.configure("Material.TText",
                       background=MaterialStyle.BACKGROUND_COLOR,
                       foreground=MaterialStyle.TEXT_COLOR,
                       padding=MaterialStyle.PADDING)

class AccountingAssistantGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Buchungsassistent")
        self.root.configure(background=MaterialStyle.BACKGROUND_COLOR)
        
        # Stil konfigurieren
        MaterialStyle.configure_styles()
        
        # Setup der Backend-Funktionalität
        self._setup_backend()
        
        # GUI-Elemente erstellen
        self._create_widgets()
        
    def _setup_backend(self):
        """Initialisiert die Backend-Funktionalität"""
        config_file = "/home/user/.aicore/config.json"
        try:
            with open(config_file, 'r') as file:
                config = json.load(file)
                for key, value in config.items():
                    os.environ[key] = str(value)
                    
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
            
        except Exception as e:
            self._show_error(f"Fehler bei der Initialisierung: {str(e)}")
    
    def _create_widgets(self):
        """Erstellt alle GUI-Elemente"""
        # Hauptframe
        self.main_frame = ttk.Frame(self.root, style="Material.TFrame")
        self.main_frame.pack(fill=tk.BOTH, expand=True, padx=20, pady=20)
        
        # Titel
        title_label = ttk.Label(self.main_frame, 
                              text="Buchungsassistent",
                              style="Material.TLabel",
                              font=('Segoe UI', 24))
        title_label.pack(pady=(0, 20))
        
        # Eingabebereich
        input_label = ttk.Label(self.main_frame, 
                              text="Eingabe Buchungsinformationen",
                              style="Material.TLabel",
                              font=('Segoe UI', 12))
        input_label.pack(anchor=tk.W)
        
        self.input_text = tk.Text(self.main_frame, 
                                height=6,
                                font=('Segoe UI', 11),
                                wrap=tk.WORD)
        self.input_text.pack(fill=tk.X, pady=(5, 20))
        
        # Ausgabebereich
        output_label = ttk.Label(self.main_frame,
                               text="Ausgabe Kontierungsinformationen",
                               style="Material.TLabel",
                               font=('Segoe UI', 12))
        output_label.pack(anchor=tk.W)
        
        self.output_text = tk.Text(self.main_frame,
                                 height=10,
                                 font=('Segoe UI', 11),
                                 wrap=tk.WORD,
                                 state='disabled')
        self.output_text.pack(fill=tk.X, pady=(5, 20))
        
        # Button-Frame
        button_frame = ttk.Frame(self.main_frame, style="Material.TFrame")
        button_frame.pack(fill=tk.X, pady=(10, 0))
        
        # Buttons
        self.start_button = tk.Button(button_frame,
                                    text="Start",
                                    command=self._process_input,
                                    bg=MaterialStyle.PRIMARY_COLOR,
                                    fg=MaterialStyle.BUTTON_TEXT_COLOR,
                                    font=('Segoe UI', 11),
                                    relief=tk.FLAT,
                                    padx=20)
        self.start_button.pack(side=tk.LEFT, padx=5)
        
        self.new_button = tk.Button(button_frame,
                                  text="Neu",
                                  command=self._clear_fields,
                                  bg=MaterialStyle.PRIMARY_COLOR,
                                  fg=MaterialStyle.BUTTON_TEXT_COLOR,
                                  font=('Segoe UI', 11),
                                  relief=tk.FLAT,
                                  padx=20)
        self.new_button.pack(side=tk.LEFT, padx=5)
        
        self.exit_button = tk.Button(button_frame,
                                   text="Verlassen",
                                   command=self._exit_application,
                                   bg=MaterialStyle.PRIMARY_COLOR,
                                   fg=MaterialStyle.BUTTON_TEXT_COLOR,
                                   font=('Segoe UI', 11),
                                   relief=tk.FLAT,
                                   padx=20)
        self.exit_button.pack(side=tk.RIGHT, padx=5)
        
    def _process_input(self):
        """Verarbeitet die Eingabe und zeigt das Ergebnis an"""
        business_case = self.input_text.get("1.0", tk.END).strip()
        if not business_case:
            self._show_error("Bitte geben Sie Buchungsinformationen ein.")
            return
            
        try:
            # B1: Buchungsinformationen extrahieren
            booking_info = self._get_booking_information(business_case)
            
            # B2: Relevante Zuordnungen abrufen
            relevant_assignments = self._retrieve_accounting_assignments(booking_info)
            
            # B3: Neue Zuordnungen erstellen
            result = self._create_accounting_assignments(booking_info, relevant_assignments)
            
            # Ergebnis anzeigen
            self.output_text.configure(state='normal')
            self.output_text.delete("1.0", tk.END)
            self.output_text.insert("1.0", result)
            self.output_text.configure(state='disabled')
            
        except Exception as e:
            self._show_error(f"Fehler bei der Verarbeitung: {str(e)}")
    
    def _get_booking_information(self, business_case: str) -> Dict:
        """Extrahiert Buchungsinformationen aus dem Geschäftsfall"""
        return {
            "business_case": business_case,
            "extracted_info": "Extrahiert aus dem Geschäftsfall"
        }
    
    def _retrieve_accounting_assignments(self, booking_info: Dict) -> List[Dict]:
        """Ruft relevante Buchungszuordnungen aus der HANA DB ab"""
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
    
    def _create_accounting_assignments(self, booking_info: Dict,
                                    relevant_assignments: List[Dict]) -> str:
        """Erstellt Buchungszuordnungen mit Hilfe des LLM"""
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
    
    def _clear_fields(self):
        """Löscht alle Eingabe- und Ausgabefelder"""
        self.input_text.delete("1.0", tk.END)
        self.output_text.configure(state='normal')
        self.output_text.delete("1.0", tk.END)
        self.output_text.configure(state='disabled')
    
    def _exit_application(self):
        """Beendet die Anwendung"""
        self.root.quit()
    
    def _show_error(self, message: str):
        """Zeigt eine Fehlermeldung im Ausgabefeld an"""
        self.output_text.configure(state='normal')
        self.output_text.delete("1.0", tk.END)
        self.output_text.insert("1.0", f"Fehler: {message}")
        self.output_text.configure(state='disabled')

def main():
    root = tk.Tk()
    app = AccountingAssistantGUI(root)
    root.mainloop()

if __name__ == "__main__":
    main() 