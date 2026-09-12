"""
data_loader.py
==============
Récupération des données de marché nécessaires au pricing :
- Spot / historique de prix du sous-jacent (yfinance)
- Taux sans risque (FRED)
- Volatilité implicite / VIX / VSTOXX (proxy)

TODO (à finaliser ensemble) :
- Confirmer le ticker exact du sous-jacent
- Choisir la source de taux (FRED series id, ex. "DGS3MO" pour le 3-mois US)
- Décider du proxy de volatilité (VIX pour S&P500, VSTOXX pour EURO STOXX 50)
  ou reconstruction depuis une chaîne d'options yfinance
"""

from dataclasses import dataclass
from datetime import date
from typing import Optional

import pandas as pd
import yfinance as yf

@dataclass
class MarketDataBundle:
    """Conteneur des données de marché brutes récupérées."""
    spot_history: pd.DataFrame          # colonnes attendues : ["Date", "Close"]
    spot_price: float                   # dernier prix disponible (S0)
    pricing_date: date
    risk_free_rate: Optional[float] = None
    dividend_yield: Optional[float] = None
    implied_vol_proxy: Optional[float] = None


def fetch_underlying_history(
    ticker: str,
    start: str,
    end: Optional[str] = None,
) -> pd.DataFrame:
    """
    Récupère l'historique de prix d'un sous-jacent via yfinance.

    Parameters
    ----------
    ticker : str
        Ticker yfinance (ex. "^STOXX50E", "^GSPC")
    start : str
        Date de début au format "YYYY-MM-DD"
    end : str, optional
        Date de fin au format "YYYY-MM-DD". Si None, jusqu'à aujourd'hui.

    Returns
    -------
    pd.DataFrame
        Historique avec au moins les colonnes ["Date", "Close"]
    """
    data = yf.download(ticker, start=start, end=end, auto_adjust=True, progress=False)
    
    if data.empty:
        raise ValueError(f"Aucune donnée récupérée pour le ticker '{ticker}'.")
    
    # yfinance renvoie parfois un MultiIndex de colonnes (Close, ticker) même
    # pour un seul ticker selon la version installée -> on aplatit si besoin.
    if isinstance(data.columns, pd.MultiIndex):
        data.columns = data.columns.get_level_values(0)

    data = data.reset_index()[["Date", "Close"]].dropna()
    data = data.sort_values("Date").reset_index(drop=True)
    return data


def fetch_risk_free_rate(series_id: str, start: str, end: Optional[str] = None) -> float:
    import pandas_datareader.data as web
    """
    Récupère un taux sans risque depuis FRED (via pandas-datareader ou API FRED directe).

    """
    end = end or date.today().strftime("%Y-%m-%d")
    series = web.DataReader(series_id, "fred", start, end).dropna()

    if series.empty:
        raise ValueError(f"Aucune donnée FRED récupérée pour la série '{series_id}'.")

    last_value = float(series.iloc[-1, 0])
    # Les séries FRED de type DGS* sont exprimées en % (ex. 4.33 pour 4.33%)
    return last_value / 100.0


def fetch_volatility_proxy(index_name: str, start: str, end: Optional[str] = None) -> float:
    """
    Récupère un proxy de volatilité implicite (VIX pour S&P 500, VSTOXX pour EURO STOXX 50).

    Returns
    -------
    float
        Volatilité annualisée en décimal (ex. 0.18 pour 18%)
    """
    data = yf.download(index_name, start=start, end=end, auto_adjust=True, progress=False)

    if data.empty:
        raise ValueError(f"Aucune donnée récupérée pour l'indice de vol '{index_name}'.")

    if isinstance(data.columns, pd.MultiIndex):
        data.columns = data.columns.get_level_values(0)

    last_value = float(data["Close"].dropna().iloc[-1])
    # Le VIX est coté directement en points de %, ex. 16.5 -> 0.165
    return last_value / 100.0


def build_market_data_bundle(config: dict) -> MarketDataBundle:
    """
    Point d'entrée principal : construit le bundle complet de données de marché
    à partir de la configuration produit.
    """
    ticker = config["underlying"]["ticker"]
    start = config.get("data", {}).get("history_start", "2015-01-01")
    end = config.get("pricing_date")  # None -> yfinance va jusqu'à aujourd'hui
    vol_ticker = config.get("data", {}).get("vol_proxy_ticker", "^VIX")
    rate_series_id = config.get("data", {}).get("risk_free_series_id", "DGS3MO")

    spot_history = fetch_underlying_history(ticker, start=start, end=end)
    spot_price = float(spot_history["Close"].iloc[-1])
    pricing_date = pd.to_datetime(spot_history["Date"].iloc[-1]).date()

    risk_free_rate = fetch_risk_free_rate(rate_series_id, start=start, end=end)
    implied_vol_proxy = fetch_volatility_proxy(vol_ticker, start=start, end=end)
    return MarketDataBundle(
        spot_history=spot_history,
        spot_price=spot_price,
        pricing_date=pricing_date,
        risk_free_rate=risk_free_rate,
        dividend_yield=None,  # TODO: brancher estimate_dividend_yield() séparément
        implied_vol_proxy=implied_vol_proxy,
    )
