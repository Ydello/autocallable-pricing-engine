"""
greeks.py
==========
Calcul des sensibilités (Greeks) du produit par la méthode bump-and-reprice :
on reprice le produit avec un léger choc sur chaque paramètre de marché,
en réutilisant le même MonteCarloEngine que pour le pricing de référence.

Greeks couverts : Delta, Gamma, Vega, Theta, Rho

TODO (à finaliser ensemble) :
- Choisir la taille des chocs (bump size) pour chaque paramètre — un compromis entre
  bruit Monte Carlo (bump trop petit) et erreur de discrétisation (bump trop grand)
- Envisager d'utiliser les MÊMES tirages aléatoires (common random numbers) pour le
  prix de base et le prix choqué, afin de réduire drastiquement le bruit sur la
  différence (c'est déjà fait ici via `random_seed` fixe)
"""

from dataclasses import dataclass, replace

import numpy as np

from product.product_specs import ProductSpecs
from simulation.gbm_simulator import GBMParameters
from pricing.monte_carlo_engine import MonteCarloEngine


@dataclass
class GreeksResult:
    delta: float
    gamma: float
    vega: float
    theta: float
    rho: float
    base_price: float


def compute_greeks(
    specs: ProductSpecs,
    market_params: GBMParameters,
    n_paths: int = 100_000,
    random_seed: int = 42,
    spot_bump_pct: float = 0.01,
    vol_bump_abs: float = 0.01,
    rate_bump_abs: float = 0.0001,
    time_bump_years: float = 1 / 365,
) -> GreeksResult:
    """
    Calcule les Greeks par différences finies centrées (sauf Theta, en différence avant).

    Utilise systématiquement le même `random_seed` pour tous les repricings (common
    random numbers), afin que la variance Monte Carlo n'affecte pas le calcul de la
    différence entre prix choqué et prix de base.

    Parameters
    ----------
    specs : ProductSpecs
    market_params : GBMParameters
        Paramètres de marché de référence (non modifiés en entrée)
    n_paths : int
        Nombre de trajectoires pour chaque repricing
    random_seed : int
        Graine fixe utilisée pour TOUS les repricings (common random numbers)
    spot_bump_pct : float
        Choc relatif sur le spot pour Delta/Gamma (ex. 0.01 = +/-1%)
    vol_bump_abs : float
        Choc absolu sur la volatilité pour Vega (ex. 0.01 = +/-1 point de vol)
    rate_bump_abs : float
        Choc absolu sur le taux pour Rho (ex. 0.0001 = +/-1bp)
    time_bump_years : float
        Choc de temps pour Theta (ex. 1/365 = 1 jour)

    Returns
    -------
    GreeksResult

    Important : `reference_spot` (le niveau S0 servant à normaliser les barrières,
    fixé à l'émission) est gelé à sa valeur de départ pour TOUS les repricings de cette
    fonction — seul `spot` (le niveau de marché courant) est choqué. C'est indispensable
    pour que Delta/Gamma soient non-nuls (cf. docstring de GBMParameters.reference_spot).
    """
    # On fige la référence dès le départ pour que les bumps de spot ne la déplacent pas.
    base_market_params = replace(market_params, reference_spot=market_params.effective_reference_spot())

    def reprice(params: GBMParameters, product_specs: ProductSpecs = specs) -> float:
        engine = MonteCarloEngine(product_specs, params)
        return engine.price(n_paths=n_paths, random_seed=random_seed).price

    base_price = reprice(base_market_params)

    # --- Delta & Gamma (bump sur le spot courant, reference_spot reste figé) ---
    spot_up = replace(base_market_params, spot=base_market_params.spot * (1 + spot_bump_pct))
    spot_down = replace(base_market_params, spot=base_market_params.spot * (1 - spot_bump_pct))
    price_up = reprice(spot_up)
    price_down = reprice(spot_down)

    h_spot = base_market_params.spot * spot_bump_pct
    delta = (price_up - price_down) / (2 * h_spot)
    gamma = (price_up - 2 * base_price + price_down) / (h_spot ** 2)

    # --- Vega (bump sur la volatilité) ---
    vol_up = replace(base_market_params, volatility=base_market_params.volatility + vol_bump_abs)
    vol_down = replace(base_market_params, volatility=base_market_params.volatility - vol_bump_abs)
    vega = (reprice(vol_up) - reprice(vol_down)) / (2 * vol_bump_abs)

    # --- Rho (bump sur le taux sans risque) ---
    rate_up = replace(base_market_params, risk_free_rate=base_market_params.risk_free_rate + rate_bump_abs)
    rate_down = replace(base_market_params, risk_free_rate=base_market_params.risk_free_rate - rate_bump_abs)
    rho = (reprice(rate_up) - reprice(rate_down)) / (2 * rate_bump_abs)

    # --- Theta (avancer le temps d'un jour : on retire la 1ère date d'observation
    # restante des specs ET on avance le spot le long d'une trajectoire à drift risque-
    # neutre pour rester cohérent, tout en gardant reference_spot figé) ---
    remaining_times = specs.observation_times[specs.observation_times > time_bump_years]
    theta_specs = replace(
        specs,
        observation_times=remaining_times - time_bump_years,
        autocall_barriers=specs.autocall_barriers[-len(remaining_times):],
        coupon_barriers=specs.coupon_barriers[-len(remaining_times):],
        maturity_years=specs.maturity_years - time_bump_years,
    )
    price_theta = reprice(base_market_params, theta_specs)
    theta = (price_theta - base_price) / time_bump_years

    return GreeksResult(
        delta=delta,
        gamma=gamma,
        vega=vega,
        theta=theta,
        rho=rho,
        base_price=base_price,
    )
