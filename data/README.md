# Data

This project downloads all data programmatically at runtime:

- **Stock prices**: 503 S&P 500 constituents via `yfinance` (457 retained after filtering)
- **Macro variables**: 5 FRED series (yield spread, VIX, unemployment, CPI, fed funds)

No static data files are stored in this folder. To reproduce, run `PythonFinance.ipynb` from the top.
