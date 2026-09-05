"""
market_calibration.py
======================
Transforme les données brutes (data_loader.py) en paramètres de modèle prêts
à l'emploi pour la simulation : volatilité annualisée, taux sans risque continu,
dividend yield.

TODO (à finaliser ensemble) :
- Choisir la méthode de calibration de la vol (historique glissante vs proxy VIX/VSTOXX
  vs implicite depuis options)
- Décider de la fenêtre de calcul de la vol historique (ex. 252 jours, 1 an, 3 ans)
- Confirmer la convention de taux (composé continu vs annuel)
"""

from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd


@dataclass
class CalibratedParameters:
    """Paramètres de marché calibrés, prêts pour le simulateur GBM."""
    spot: float
    risk_free_rate: float       # taux continu (convention utilisée dans le GBM sous Q)
    dividend_yield: float
    volatility: float           # volatilité annualisée


def historical_volatility(price_history: pd.DataFrame, window_days: int = 252) -> float:
    """
    Calcule la volatilité historique annualisée à partir des rendements log.

    Parameters
    ----------
    price_history : pd.DataFrame
        Doit contenir une colonne "Close"
    window_days : int
        Fenêtre de calcul en jours de trading (défaut 252 = 1 an)

    Returns
    -------
    float
        Volatilité annualisée (décimal)

    TODO : valider le choix de la fenêtre avec les données réelles une fois chargées
    """
    prices = price_history["Close"].tail(window_days + 1)
    log_returns = np.log(prices / prices.shift(1)).dropna()
    daily_vol = log_returns.std()
    annualized_vol = daily_vol * np.sqrt(252)
    return float(annualized_vol)


def annual_rate_to_continuous(annual_rate: float) -> float:
    """
    Convertit un taux annuel composé en taux continu équivalent : r_c = ln(1 + r_annual).

    TODO : confirmer avec quelle convention la source de taux (FRED) exprime ses taux
           (souvent déjà en taux simple annualisé -> à adapter si besoin)
    """
    return float(np.log(1 + annual_rate))


def estimate_dividend_yield(index_name: str) -> float:
    """
    Estime le dividend yield de l'indice.

    TODO : décider de la source (donnée publiée par le fournisseur d'indice,
           ou approximation via l'écart entre futures et spot - "implied repo")
    """
    raise NotImplementedError("TODO: définir la source du dividend yield.")


def calibrate(
    spot: float,
    price_history: pd.DataFrame,
    raw_risk_free_rate: Optional[float] = None,
    raw_volatility_proxy: Optional[float] = None,
    dividend_yield: Optional[float] = None,
    vol_window_days: int = 252,
) -> CalibratedParameters:
    """
    Point d'entrée principal de calibration.

    Si `raw_volatility_proxy` est fourni (ex. VIX/VSTOXX), il est utilisé directement.
    Sinon, on retombe sur la volatilité historique calculée depuis `price_history`.

    TODO : arbitrer la logique de priorité entre vol historique et vol implicite
           une fois les deux sources disponibles (comparer, choisir, ou moyenne pondérée)
    """
    vol = raw_volatility_proxy if raw_volatility_proxy is not None else historical_volatility(
        price_history, window_days=vol_window_days
    )

    if raw_risk_free_rate is None:
        raise NotImplementedError("TODO: taux sans risque manquant — brancher data_loader.py")

    if dividend_yield is None:
        raise NotImplementedError("TODO: dividend yield manquant — brancher data_loader.py")

    return CalibratedParameters(
        spot=spot,
        risk_free_rate=raw_risk_free_rate,
        dividend_yield=dividend_yield,
        volatility=vol,
    )
