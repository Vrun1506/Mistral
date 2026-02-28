"""
Central configuration — loaded from .env via python-dotenv.
All other modules import constants from here instead of defining their own.
"""

import os
from pathlib import Path

from dotenv import load_dotenv

_env_path = Path(__file__).parent / ".env"
load_dotenv(_env_path)

MISTRAL_API_KEY: str = os.environ.get("MISTRAL_API_KEY", "")
MISTRAL_MODEL: str = os.environ.get("MISTRAL_MODEL", "mistral-large-latest")
MCP_SERVER_URL: str = os.environ.get("MCP_SERVER_URL", "https://mcp.notion.com")
OAUTH_CALLBACK_PORT: int = int(os.environ.get("OAUTH_CALLBACK_PORT", "8789"))
REDIRECT_URI: str = f"http://localhost:{OAUTH_CALLBACK_PORT}/callback"

PROJECT_DIR: Path = Path(__file__).parent
TOKEN_FILE: Path = PROJECT_DIR / "tokens.json"
CLIENT_FILE: Path = PROJECT_DIR / "client.json"
