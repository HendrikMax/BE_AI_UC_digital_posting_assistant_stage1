# InputManager.py
# This module contains the InputManager class to avoid circular imports.

import json
from IPython.display import display, HTML

class InputManager:
    def __init__(self):
        self.input_text = ""
        self.history = []

    def get_current_input(self):
        return self.input_text

    def get_history(self):
        return self.history

    def save_history(self, filename="input_history.json"):
        with open(filename, "w", encoding="utf-8") as f:
            json.dump(self.history, f, ensure_ascii=False, indent=2)

    def load_history(self, filename="input_history.json"):
        try:
            with open(filename, "r", encoding="utf-8") as f:
                self.history = json.load(f)
        except FileNotFoundError:
            print("Keine gespeicherte Historie gefunden.")
    
    def display_widget(self):
        display(HTML("<h1>Digitaler Buchungsassistent - Kontierungshilfe</h1>"))
    
    def update_output(self, output):
        """
        Aktualisiert die Ausgabe basierend auf der Antwort des Systems.

        Args:
            output (str): Die generierte Antwort.
        """
        self.output = output
        print(f"Aktualisierte Ausgabe: {output}")
