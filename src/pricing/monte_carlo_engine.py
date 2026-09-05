"""
monte_carlo_engine.py
======================
Moteur de pricing principal : orchestre la simulation des trajectoires,
le calcul du payoff Phoenix, et l'actualisation, pour produire le prix
et son intervalle de confiance.

Ce moteur est réutilisé tel quel par :
- Le pricing de référence (notebooks/03_validation_pricing.ipynb)
- Le calcul des Greeks par bump-and-reprice (risk/greeks.py)
- Les stress tests (risk/stress_testing.py)

TODO (à finaliser ensemble) :
- Brancher continuous_min_ratio si le knock-in doit être observé en continu
"""

from dataclasses import dataclass

import numpy as np

from product.product_specs import ProductSpecs
from product.autocall_payoff import compute_phoenix_payoff
from simulation.gbm_simulator import GBMParameters, simulate_gbm_paths
from utils.discounting import discount_single_payment


@dataclass
class PricingResult:
    price: float
    std_error: float
    confidence_interval_95: tuple[float, float]
    n_paths: int
    autocall_probability_by_date: np.ndarray   # (n_observations,) proba d'autocall à chaque date
    knock_in_probability: float
    running_means: np.ndarray                   # pour analyse de convergence


class MonteCarloEngine:
    """Moteur de pricing Monte Carlo pour le Phoenix Autocall."""

    def __init__(self, specs: ProductSpecs, market_params: GBMParameters):
        specs.validate()
        self.specs = specs
        self.market_params = market_params

    def price(
        self,
        n_paths: int = 100_000,
        random_seed: int | None = 42,
        antithetic: bool = True,
    ) -> PricingResult:
        """
        Calcule le prix du produit par Monte Carlo.

        Parameters
        ----------
        n_paths : int
            Nombre de trajectoires simulées
        random_seed : int, optional
            Graine aléatoire pour reproductibilité
        antithetic : bool
            Active la réduction de variance par variables antithétiques

        Returns
        -------
        PricingResult
        """
        paths = simulate_gbm_paths(
            self.market_params,
            self.specs.observation_times,
            n_paths=n_paths,
            random_seed=random_seed,
            antithetic=antithetic,
        )

        payoff_result = compute_phoenix_payoff(
            paths=paths,
            specs=self.specs,
            spot_initial=self.market_params.effective_reference_spot(),
        )

        discounted_payoffs = discount_single_payment(
            payoff_result.total_payoff,
            payoff_result.payment_times,
            self.market_params.risk_free_rate,
        )

        price = float(np.mean(discounted_payoffs))
        std_error = float(np.std(discounted_payoffs, ddof=1) / np.sqrt(n_paths))
        ci_95 = (price - 1.96 * std_error, price + 1.96 * std_error)

        # Probabilité d'autocall à chaque date (utile pour la restitution)
        ref_spot = self.market_params.effective_reference_spot()
        autocall_matrix = paths / ref_spot >= self.specs.autocall_barriers[np.newaxis, :]
        # ne compter que la première occurrence par trajectoire
        first_true = np.zeros_like(autocall_matrix, dtype=bool)
        has_any = autocall_matrix.any(axis=1)
        first_idx = np.where(has_any, autocall_matrix.argmax(axis=1), -1)
        valid = first_idx >= 0
        first_true[np.arange(len(first_idx))[valid], first_idx[valid]] = True
        autocall_prob_by_date = first_true.mean(axis=0)

        knock_in_probability = float(payoff_result.knocked_in.mean())

        running_means = np.cumsum(discounted_payoffs) / np.arange(1, n_paths + 1)

        return PricingResult(
            price=price,
            std_error=std_error,
            confidence_interval_95=ci_95,
            n_paths=n_paths,
            autocall_probability_by_date=autocall_prob_by_date,
            knock_in_probability=knock_in_probability,
            running_means=running_means,
        )
