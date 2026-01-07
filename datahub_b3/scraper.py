"""B3 Index Scraper - Coleta composição de índices via API da B3."""
import base64
import json
import logging
from dataclasses import dataclass
from datetime import date
from typing import Optional
import httpx

logger = logging.getLogger(__name__)


@dataclass
class IndexComponent:
    """Componente de um índice."""
    symbol: str
    company_name: str
    stock_type: str
    participation_pct: float
    theoretical_qty: int


@dataclass
class IndexData:
    """Dados completos de um índice."""
    index_code: str
    date: date
    total_components: int
    reductor: float
    theoretical_qty: int
    components: list[IndexComponent]


def _build_api_url(index_code: str, page_size: int = 200) -> str:
    """Constrói URL da API da B3 com payload base64."""
    payload = {
        "language": "pt-br",
        "pageNumber": 1,
        "pageSize": page_size,
        "index": index_code
    }
    encoded = base64.b64encode(json.dumps(payload).encode()).decode()
    return f"https://sistemaswebb3-listados.b3.com.br/indexProxy/indexCall/GetPortfolioDay/{encoded}"


def _parse_number(value: str) -> float:
    """Parse número no formato brasileiro (1.234,56 -> 1234.56)."""
    if not value:
        return 0.0
    return float(value.replace(".", "").replace(",", "."))


def fetch_index(index_code: str, timeout: float = 30.0) -> Optional[IndexData]:
    """Busca composição de um índice da B3."""
    url = _build_api_url(index_code)
    logger.info(f"Fetching {index_code} from B3 API...")
    
    try:
        with httpx.Client(timeout=timeout) as client:
            resp = client.get(url, headers={"Accept": "application/json"})
            resp.raise_for_status()
            data = resp.json()
    except Exception as e:
        logger.error(f"Failed to fetch {index_code}: {e}")
        return None
    
    if not data.get("results"):
        logger.warning(f"No data for {index_code}")
        return None
    
    header = data.get("header", {})
    page = data.get("page", {})
    
    # Parse date (format: DD/MM/YY)
    date_str = header.get("date", "")
    try:
        day, month, year = date_str.split("/")
        index_date = date(2000 + int(year), int(month), int(day))
    except:
        index_date = date.today()
    
    components = []
    for item in data["results"]:
        comp = IndexComponent(
            symbol=item["cod"],
            company_name=item["asset"],
            stock_type=item["type"],
            participation_pct=_parse_number(item["part"]),
            theoretical_qty=int(_parse_number(item.get("theoricalQty", "0")))
        )
        components.append(comp)
    
    return IndexData(
        index_code=index_code,
        date=index_date,
        total_components=page.get("totalRecords", len(components)),
        reductor=_parse_number(header.get("reductor", "0")),
        theoretical_qty=int(_parse_number(header.get("theoricalQty", "0"))),
        components=components
    )


def fetch_all_indices(indices: list[str] = None) -> dict[str, IndexData]:
    """Busca múltiplos índices."""
    from .config import B3_INDICES
    
    if indices is None:
        indices = list(B3_INDICES.keys())
    
    results = {}
    for code in indices:
        data = fetch_index(code)
        if data:
            results[code] = data
            logger.info(f"✓ {code}: {data.total_components} components")
    
    return results








































