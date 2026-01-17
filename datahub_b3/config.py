"""Configuration for DataHub B3."""
import os
from pathlib import Path

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

PROJECT_ROOT = Path(__file__).parent.parent
DATABASE_URL = os.getenv("DATABASE_URL", "")

# B3 API endpoints (unofficial - from browser scraping)
B3_INDEX_API_BASE = "https://sistemaswebb3-listados.b3.com.br/indexProxy/indexCall"

# Available indices
B3_INDICES = {
    "IBOV": {"name": "Índice Bovespa", "desc": "Principal indicador do mercado brasileiro"},
    "IBRA": {"name": "Índice Brasil Amplo", "desc": "99% do somatório de negociabilidade"},
    "IBRX100": {"name": "IBrX 100", "desc": "100 ações mais negociadas"},
    "SMLL": {"name": "Índice Small Cap", "desc": "Ações de menor capitalização"},
    "IDIV": {"name": "Índice Dividendos", "desc": "Ações com maiores dividendos"},
    "IFIX": {"name": "Índice de FIIs", "desc": "Fundos imobiliários"},
}

# Schedule (Brasília timezone - UTC-3)
UPDATE_TIME = "18:30"  # After market close
TIMEZONE = "America/Sao_Paulo"









































