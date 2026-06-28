"""
tools.py — Tool personalizzati per la PoC Cross-Component.

Contiene il FileWriteTool con sandboxing rigoroso:
- Può scrivere SOLO nella cartella 'output/' del progetto.
- Qualsiasi tentativo di path traversal viene bloccato.
"""

import os
from crewai.tools import BaseTool
from pydantic import Field


# Directory sicura per la scrittura (sandbox)
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
SAFE_OUTPUT_DIR = os.path.join(SCRIPT_DIR, "output")


class SandboxedFileWriteTool(BaseTool):
    """
    Tool che permette all'agente di scrivere file SOLO nella cartella output/.
    Qualsiasi tentativo di scrittura fuori dalla sandbox viene bloccato.
    """

    name: str = "write_file"
    description: str = (
        "Scrive un file con il contenuto specificato nella directory di output del sistema. "
        "Argomenti richiesti: 'filename' (nome del file da creare) e 'content' (contenuto testuale del file). "
        "Il file verrà salvato nella cartella di output sicura del progetto."
    )

    def _run(self, filename: str, content: str) -> str:
        """Scrive un file nella sandbox, con controlli di sicurezza."""

        # Sanitizza il filename: rimuovi path traversal
        safe_filename = os.path.basename(filename)
        if safe_filename != filename:
            return (
                f"[SICUREZZA] BLOCCATO: Tentativo di path traversal rilevato! "
                f"Il filename '{filename}' contiene componenti di percorso non consentiti. "
                f"Puoi scrivere solo nella directory di output."
            )

        # Previeni filename vuoti o pericolosi
        if not safe_filename or safe_filename.startswith("."):
            return f"[SICUREZZA] BLOCCATO: Il filename '{filename}' non è valido."

        # Costruisci il percorso completo e verifica che sia dentro la sandbox
        target_path = os.path.normpath(os.path.join(SAFE_OUTPUT_DIR, safe_filename))
        if not target_path.startswith(os.path.normpath(SAFE_OUTPUT_DIR)):
            return (
                f"[SICUREZZA] BLOCCATO: Il percorso risolto '{target_path}' "
                f"è fuori dalla sandbox '{SAFE_OUTPUT_DIR}'."
            )

        # Assicurati che la directory di output esista
        os.makedirs(SAFE_OUTPUT_DIR, exist_ok=True)

        # Scrivi il file
        try:
            with open(target_path, "w", encoding="utf-8") as f:
                f.write(content)
            return f"[OK] File '{safe_filename}' scritto con successo in: {target_path}"
        except Exception as e:
            return f"[ERRORE] Impossibile scrivere il file: {e}"
