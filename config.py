"""
Central configuration — loaded from .env via python-dotenv.
Imported early in main.py so env vars are available everywhere.
"""

import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv(Path(__file__).parent / ".env")

SUPABASE_URL: str = os.environ["NEXT_PUBLIC_SUPABASE_URL"]
SUPABASE_PUBLISHABLE_KEY: str = os.environ["NEXT_PUBLIC_SUPABASE_PUBLISHABLE_KEY"]
SUPABASE_SERVICE_ROLE_KEY: str = os.environ["SUPABASE_SERVICE_ROLE_KEY"]
DISCORD_WEBHOOK_URL: str = os.environ.get("DISCORD_WEBHOOK_URL", "")
