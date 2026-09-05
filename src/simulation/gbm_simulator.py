"""
gbm_simulator.py
================
Génère des trajectoires du sous-jacent selon un mouvement brownien géométrique (GBM)
sous la mesure risque-neutre :

    dS_t = (r - q) S_t dt + sigma S_t dW_t

Discrétisation exacte (schéma log-normal, pas d'approximation d'Euler nécessaire
pour un GBM à volatilité constante) :

    S_{t+dt} = S_t * exp( (r - q - 0.5*sigma^2)*dt + sigma*sqrt(dt)*Z ),  Z ~ N(0,1)

TODO (à finaliser ensemble) :
- Si on passe à un modèle plus riche (Heston, vol locale), ce module devra être
  étendu ou remplacé par un schéma d'Euler/Milstein adapté
"""

from dataclasses import dataclass
from typing import Optional

import numpy as np


@dataclass
class GBMParameters:
    spot: float
    risk_free_rate: float
    dividend_yield: float
    volatility: float
    reference_spot: Optional[float] = None
    """
    Niveau de référence (S0) utilisé pour normaliser les barrières du produit
    (fixé au moment de l'émission et gelé ensuite). Si None, on suppose qu'on price
    à l'émission et reference_spot = spot.

    Distinction importante pour le calcul des Greeks : quand on "bump" le spot pour
    calculer Delta/Gamma, il faut déplacer `spot` (le niveau de marché courant simulé)
    SANS déplacer `reference_spot` (les barrières sont fixées contractuellement et ne
    bougent pas quand le marché bouge après l'émission). Sans cette séparation, un choc
    simultané des deux valeurs s'annule dans les ratios S_t/S0 utilisés par les barrières,
    ce qui donnerait un Delta et un Theta artificiellement nuls.
    """

    def effective_reference_spot(self) -> float:
        return self.reference_spot if self.reference_spot is not None else self.spot


def simulate_gbm_paths(
    params: GBMParameters,
    observation_times: np.ndarray,
    n_paths: int,
    random_seed: Optional[int] = None,
    antithetic: bool = False,
) -> np.ndarray:
    """
    Simule des trajectoires GBM aux dates d'observation spécifiées.

    Parameters
    ----------
    params : GBMParameters
        Paramètres du modèle (spot, taux, dividende, volatilité)
    observation_times : np.ndarray
        Dates d'observation en années depuis aujourd'hui, ex. [0.25, 0.5, 0.75, 1.0, ...]
        (doit être triée par ordre croissant)
    n_paths : int
        Nombre de trajectoires à simuler
    random_seed : int, optional
        Graine aléatoire pour la reproductibilité
    antithetic : bool
        Si True, génère des paires de trajectoires antithétiques (Z et -Z)
        pour réduire la variance. Dans ce cas, n_paths doit être pair ;
        la moitié des chocs est générée puis dupliquée en négatif.

    Returns
    -------
    np.ndarray
        Tableau de forme (n_paths, n_observations) contenant S_t à chaque date
        d'observation, pour chaque trajectoire.
    """
    if random_seed is not None:
        rng = np.random.default_rng(random_seed)
    else:
        rng = np.random.default_rng()

    observation_times = np.asarray(observation_times, dtype=float)
    n_obs = len(observation_times)

    # dt entre chaque date d'observation (le premier dt est depuis t=0)
    dt = np.diff(np.concatenate(([0.0], observation_times)))
    if np.any(dt <= 0):
        raise ValueError("observation_times doit être strictement croissant et positif.")

    drift = params.risk_free_rate - params.dividend_yield - 0.5 * params.volatility ** 2

    if antithetic:
        if n_paths % 2 != 0:
            raise ValueError("n_paths doit être pair pour la méthode antithétique.")
        half = n_paths // 2
        z_half = rng.standard_normal(size=(half, n_obs))
        z = np.concatenate([z_half, -z_half], axis=0)
    else:
        z = rng.standard_normal(size=(n_paths, n_obs))

    # increments log-normaux cumulés le long du temps
    log_increments = drift * dt[np.newaxis, :] + params.volatility * np.sqrt(dt)[np.newaxis, :] * z
    log_paths = np.cumsum(log_increments, axis=1)

    paths = params.spot * np.exp(log_paths)
    return paths


def simulate_gbm_full_paths(
    params: GBMParameters,
    total_time: float,
    n_steps: int,
    n_paths: int,
    random_seed: Optional[int] = None,
) -> np.ndarray:
    """
    Simule des trajectoires GBM sur une grille de temps fine et régulière
    (utile si on a besoin d'une observation continue, ex. knock-in "American-style").

    Returns
    -------
    np.ndarray
        Tableau de forme (n_paths, n_steps + 1), incluant S_0 en première colonne.

    TODO : à activer si la structure retenue utilise un knock-in observé en continu
           plutôt qu'à des dates discrètes (cf. product_config.yaml -> observation_type)
    """
    dt = total_time / n_steps
    rng = np.random.default_rng(random_seed) if random_seed is not None else np.random.default_rng()

    drift = params.risk_free_rate - params.dividend_yield - 0.5 * params.volatility ** 2
    z = rng.standard_normal(size=(n_paths, n_steps))

    log_increments = drift * dt + params.volatility * np.sqrt(dt) * z
    log_paths = np.cumsum(log_increments, axis=1)

    paths = params.spot * np.exp(log_paths)
    initial_col = np.full((n_paths, 1), params.spot)
    return np.concatenate([initial_col, paths], axis=1)
