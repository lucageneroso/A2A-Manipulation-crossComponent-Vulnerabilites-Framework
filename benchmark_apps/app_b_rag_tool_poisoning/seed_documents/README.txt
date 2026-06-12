This document is intentionally empty. The adversarial document is seeded
directly in app.py at startup (see the initialize_vector_store function).

The adversarial payload is:
  - Semantically close to "employee" queries (to ensure retrieval)
  - Contains an indirect prompt injection that triggers SQL tool misuse

For research transparency, the exact adversarial document text is logged
at application startup with the prefix "[APP-B] Vector store seeded".
