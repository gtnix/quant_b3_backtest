from pathlib import Path


def pytest_addoption(parser):
    group = parser.getgroup("html_exec_validation")
    group.addoption(
        "--report-html",
        action="store",
        default=str(Path("reports") / "portfolio_execution_report.html"),
        help="Path to the generated backtest HTML report.",
    )
    group.addoption(
        "--report-csv",
        action="store",
        default=str(Path("reports") / "fuzzy_fajuto_execution_history.csv"),
        help="Optional path to execution history CSV fallback.",
    )
    group.addoption(
        "--tolerance",
        action="store",
        type=float,
        default=0.01,
        help="Absolute price tolerance in BRL (default 0.01).",
    )

