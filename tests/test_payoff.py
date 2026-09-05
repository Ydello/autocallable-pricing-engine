"""
test_payoff.py
================
Tests unitaires pour la logique de payoff Phoenix Autocall.
Utilise des trajectoires construites à la main (pas de simulation aléatoire)
pour vérifier précisément chaque branche de la logique (autocall, mémoire, knock-in).
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import numpy as np
import pytest

from product.product_specs import ProductSpecs
from product.autocall_payoff import compute_phoenix_payoff


@pytest.fixture
def simple_specs():
    """Produit simplifié à 4 dates d'observation pour des tests lisibles."""
    return ProductSpecs(
        nominal=1000.0,
        maturity_years=1.0,
        observation_times=np.array([0.25, 0.5, 0.75, 1.0]),
        autocall_barriers=np.array([1.00, 1.00, 1.00, 1.00]),
        coupon_barriers=np.array([0.70, 0.70, 0.70, 0.70]),
        coupon_rate_period=0.0225,  # 2.25% par trimestre = 9%/an
        memory_effect=True,
        knock_in_barrier=0.60,
        knock_in_observation="at_maturity",
    )


def test_immediate_autocall(simple_specs):
    """Le prix dépasse la barrière dès la première observation -> autocall immédiat."""
    paths = np.array([[105.0, 999.0, 999.0, 999.0]])  # S0 implicite = 100
    result = compute_phoenix_payoff(paths, simple_specs, spot_initial=100.0)

    assert result.autocalled[0]
    # Capital + 1 coupon versés à la première date, rien après
    assert result.undiscounted_cashflows[0, 0] == pytest.approx(1000.0 + 22.5)
    assert result.undiscounted_cashflows[0, 1:].sum() == 0.0
    assert result.payment_times[0] == 0.25


def test_never_autocall_no_knock_in(simple_specs):
    """Le prix reste toujours entre les deux barrières -> pas d'autocall, pas de knock-in."""
    paths = np.array([[85.0, 85.0, 85.0, 85.0]])  # au-dessus de coupon (70) sous autocall (100)
    result = compute_phoenix_payoff(paths, simple_specs, spot_initial=100.0)

    assert not result.autocalled[0]
    assert not result.knocked_in[0]
    # 4 coupons simples versés (pas de rattrapage nécessaire, chaque date est éligible)
    expected_coupons = 4 * 22.5
    expected_capital = 1000.0  # capital protégé, pas de knock-in
    assert result.total_payoff[0] == pytest.approx(expected_coupons + expected_capital)


def test_memory_effect_catches_up_missed_coupon(simple_specs):
    """
    Trimestre 1 : sous la barrière de coupon -> rien
    Trimestre 2 : au-dessus -> devrait rattraper le coupon manqué du trimestre 1 + le sien
    """
    paths = np.array([[60.0, 90.0, 90.0, 90.0]])  # T1: sous coupon barrier, T2+: au-dessus
    result = compute_phoenix_payoff(paths, simple_specs, spot_initial=100.0)

    assert result.undiscounted_cashflows[0, 0] == 0.0  # rien au trimestre 1
    # Au trimestre 2 : rattrapage de 2 périodes (T1 manqué + T2)
    assert result.undiscounted_cashflows[0, 1] == pytest.approx(2 * 22.5)


def test_knock_in_reduces_capital_at_maturity(simple_specs):
    """Le prix touche la barrière knock-in à maturité -> perte en capital proportionnelle."""
    paths = np.array([[60.0, 60.0, 60.0, 50.0]])  # jamais autocall, sous coupon barrier partout, KI à maturité
    result = compute_phoenix_payoff(paths, simple_specs, spot_initial=100.0)

    assert not result.autocalled[0]
    assert result.knocked_in[0]
    # Capital remboursé = nominal * ratio final = 1000 * 0.50
    final_capital = result.undiscounted_cashflows[0, -1]
    assert final_capital == pytest.approx(500.0)


def test_no_knock_in_if_recovers_by_maturity(simple_specs):
    """Le prix descend bas puis remonte au-dessus du knock-in à maturité (observation at_maturity)."""
    paths = np.array([[40.0, 40.0, 40.0, 95.0]])  # très bas puis remonte à 95 à maturité
    result = compute_phoenix_payoff(paths, simple_specs, spot_initial=100.0)

    # observation "at_maturity" : seul le niveau final compte pour le knock-in
    assert not result.knocked_in[0]
    # La dernière cellule contient capital (protégé, pas de KI) + coupon de la dernière
    # période avec rattrapage mémoire des 4 trimestres (jamais éligible avant, tous < 70%)
    expected_cell = 1000.0 + 4 * 22.5
    final_cell = result.undiscounted_cashflows[0, -1]
    assert final_cell == pytest.approx(expected_cell)


def test_vectorized_multiple_paths(simple_specs):
    """Vérifie que plusieurs trajectoires sont traitées correctement en parallèle."""
    paths = np.array([
        [105.0, 999.0, 999.0, 999.0],   # autocall immédiat
        [85.0, 85.0, 85.0, 85.0],        # jamais autocall, jamais KI
        [60.0, 60.0, 60.0, 50.0],        # jamais autocall, KI à maturité
    ])
    result = compute_phoenix_payoff(paths, simple_specs, spot_initial=100.0)

    assert result.autocalled.tolist() == [True, False, False]
    assert result.knocked_in.tolist() == [False, False, True]
    assert len(result.total_payoff) == 3
