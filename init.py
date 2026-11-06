#!/usr/bin/env python3
"""
Project Overpass — one-step setup & launch.

    python init.py            # opens the setup page in your browser
    python init.py --no-browser --port 9000

The page walks through: OpenAI API key -> dependency install + pip-audit ->
run the API, the Streamlit frontend, or the pytest suite.
Uses only the standard library, so it works before anything is installed.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from launcher.server import main  # noqa: E402

if __name__ == "__main__":
    raise SystemExit(main())
