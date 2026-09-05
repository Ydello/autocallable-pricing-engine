"""
discounting.py
===============
Fonctions d'actualisation des flux de trésorerie.

TODO (à finaliser ensemble) :
- Si on calibre une courbe de taux (plutôt qu'un taux plat), remplacer
  discount_factor par une interpolation sur la courbe (ex. via scipy.interpolate)
"""

import numpy as np


def discount_factor(rate: float, time: np.ndarray | float) -> np.ndarray | float:
    """
    Facteur d'actualisation à taux continu constant : DF(t) = exp(-r * t)

    Parameters
    ----------
    rate : float
        Taux sans risque continu
    time : float ou np.ndarray
        Horizon(s) de temps en années

    Returns
    -------
    float ou np.ndarray
        Facteur(s) d'actualisation
    """
    return np.exp(-rate * np.asarray(time))


def discount_cashflows_by_row(
    cashflows: np.ndarray,
    observation_times: np.ndarray,
    rate: float,
) -> np.ndarray:
    """
    Actualise une matrice de flux (n_paths, n_observations) où chaque colonne j
    correspond au flux payé à observation_times[j], puis somme par ligne.

    Returns
    -------
    np.ndarray
        Valeur actualisée totale par trajectoire, forme (n_paths,)
    """
    dfs = discount_factor(rate, observation_times)  # (n_observations,)
    discounted = cashflows * dfs[np.newaxis, :]
    return discounted.sum(axis=1)


def discount_single_payment(amount: np.ndarray, payment_time: np.ndarray, rate: float) -> np.ndarray:
    """
    Actualise un flux unique par trajectoire (utile quand on a déjà agrégé
    le payoff total par trajectoire avec sa date de paiement associée,
    ex. PayoffResult.total_payoff / payment_times).
    """
    return amount * discount_factor(rate, payment_time)
