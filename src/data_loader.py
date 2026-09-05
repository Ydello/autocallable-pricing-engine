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

    TODO : implémenter l'appel yfinance réel (import yfinance as yf ; yf.download(...))
           puis nettoyer (valeurs manquantes, jours fériés, etc.)
    """
    raise NotImplementedError(
        "TODO: implémenter la récupération yfinance. "
        "Squelette prêt, à remplir ensemble avec le ticker définitif."
    )


def fetch_risk_free_rate(series_id: str, start: str, end: Optional[str] = None) -> float:
    """
    Récupère un taux sans risque depuis FRED (via pandas-datareader ou API FRED directe).

    Parameters
    ----------
    series_id : str
        Identifiant de la série FRED (ex. "DGS3MO", "DGS1", "ECBESTRVOLWGTTRMDMNRT")

    Returns
    -------
    float
        Dernier taux disponible sur la période, en décimal (ex. 0.035 pour 3.5%)

    TODO : implémenter l'appel FRED réel + choisir la maturité de taux
           cohérente avec l'horizon du produit (3 ans -> taux 3 ans si dispo)
    """
    raise NotImplementedError("TODO: implémenter l'accès FRED.")


def fetch_volatility_proxy(index_name: str, start: str, end: Optional[str] = None) -> float:
    """
    Récupère un proxy de volatilité implicite (VIX pour S&P 500, VSTOXX pour EURO STOXX 50).

    Returns
    -------
    float
        Volatilité annualisée en décimal (ex. 0.18 pour 18%)

    TODO : décider si on utilise directement l'indice de vol (simple mais approximatif)
           ou si on reconstruit une vol implicite depuis une chaîne d'options
           (plus précis mais plus complexe — cf. market_calibration.py)
    """
    raise NotImplementedError("TODO: implémenter la récupération du proxy de volatilité.")


def build_market_data_bundle(config: dict) -> MarketDataBundle:
    """
    Point d'entrée principal : construit le bundle complet de données de marché
    à partir de la configuration produit.

    TODO : orchestrer les trois fonctions ci-dessus une fois chacune finalisée.
    """
    raise NotImplementedError("TODO: assembler le bundle une fois les sous-fonctions prêtes.")
