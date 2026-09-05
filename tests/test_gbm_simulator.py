"""
test_gbm_simulator.py
=======================
Tests unitaires pour la génération de trajectoires GBM.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import numpy as np
import pytest

from simulation.gbm_simulator import GBMParameters, simulate_gbm_paths


@pytest.fixture
def default_params():
    return GBMParameters(spot=100.0, risk_free_rate=0.03, dividend_yield=0.01, volatility=0.20)


def test_output_shape(default_params):
    obs_times = np.array([0.25, 0.5, 0.75, 1.0])
    paths = simulate_gbm_paths(default_params, obs_times, n_paths=1000, random_seed=1)
    assert paths.shape == (1000, 4)


def test_all_prices_positive(default_params):
    obs_times = np.array([0.25, 0.5, 1.0, 2.0, 3.0])
    paths = simulate_gbm_paths(default_params, obs_times, n_paths=5000, random_seed=1)
    assert np.all(paths > 0)


def test_martingale_property_approx(default_params):
    """
    Sous la mesure risque-neutre, E[S_T * exp(-(r-q)*T)] doit être proche de S0
    (propriété de martingale du prix actualisé/forward). On tolère un écart lié
    au bruit Monte Carlo.
    """
    obs_times = np.array([1.0])
    n_paths = 200_000
    paths = simulate_gbm_paths(default_params, obs_times, n_paths=n_paths, random_seed=1, antithetic=True)

    forward_price = default_params.spot * np.exp(
        (default_params.risk_free_rate - default_params.dividend_yield) * 1.0
    )
    simulated_mean = paths[:, 0].mean()

    relative_error = abs(simulated_mean - forward_price) / forward_price
    assert relative_error < 0.01, f"Erreur relative trop grande : {relative_error:.4f}"


def test_antithetic_requires_even_paths(default_params):
    obs_times = np.array([1.0])
    with pytest.raises(ValueError):
        simulate_gbm_paths(default_params, obs_times, n_paths=999, antithetic=True)


def test_reproducibility_with_seed(default_params):
    obs_times = np.array([0.5, 1.0])
    paths_1 = simulate_gbm_paths(default_params, obs_times, n_paths=100, random_seed=123)
    paths_2 = simulate_gbm_paths(default_params, obs_times, n_paths=100, random_seed=123)
    np.testing.assert_array_equal(paths_1, paths_2)
