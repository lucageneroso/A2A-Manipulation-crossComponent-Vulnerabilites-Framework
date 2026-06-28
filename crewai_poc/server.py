"""
server.py — Web server locale per la PoC Cross-Component.

Espone la cartella 'www/' su http://127.0.0.1:8000
Rappresenta un sito esterno che contiene un payload IAF nascosto.
Il server è vincolato SOLO a localhost (127.0.0.1) per sicurezza.
"""

import http.server
import socketserver
import os
import sys

PORT = 8000
BIND_ADDRESS = "127.0.0.1"  # Solo localhost, non esposto su rete

# Cambia la directory di lavoro nella cartella www/
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
WWW_DIR = os.path.join(SCRIPT_DIR, "www")

if not os.path.isdir(WWW_DIR):
    print(f"[ERRORE] La cartella '{WWW_DIR}' non esiste. Creala prima di avviare il server.")
    sys.exit(1)

os.chdir(WWW_DIR)


class QuietHandler(http.server.SimpleHTTPRequestHandler):
    """Handler che logga le richieste in modo leggibile."""

    def log_message(self, format, *args):
        print(f"[SERVER] {self.client_address[0]} - {args[0]}")


def main():
    with socketserver.TCPServer((BIND_ADDRESS, PORT), QuietHandler) as httpd:
        print("=" * 60)
        print(f"  SERVER MALEVOLO ATTIVO")
        print(f"  URL: http://{BIND_ADDRESS}:{PORT}")
        print(f"  Serving directory: {WWW_DIR}")
        print(f"  Premi Ctrl+C per fermare il server")
        print("=" * 60)
        try:
            httpd.serve_forever()
        except KeyboardInterrupt:
            print("\n[SERVER] Spento.")
            httpd.server_close()


if __name__ == "__main__":
    main()
