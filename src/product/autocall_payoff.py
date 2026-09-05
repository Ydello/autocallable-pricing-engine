"""
autocall_payoff.py
===================
Implémente la formalisation mathématique du payoff Phoenix Autocall :

    - Temps d'autocall tau = première date où S_i / S0 >= barrière d'autocall
    - Coupon avec effet mémoire (rattrapage des coupons manqués)
    - Payoff terminal avec knock-in sur le capital si jamais autocallé

Le calcul est vectorisé sur l'ensemble des trajectoires simulées (pas de boucle
Python trajectoire par trajectoire, pour la performance avec n_paths >= 100k).

TODO (à finaliser ensemble) :
- Confirmer si le knock-in doit être observé en continu (nécessite les trajectoires
  complètes, pas seulement aux dates d'observation -> cf. simulate_gbm_full_paths)
"""

from dataclasses import dataclass

import numpy as np

from product.product_specs import ProductSpecs


@dataclass
class PayoffResult:
    """Résultat du calcul de payoff, trajectoire par trajectoire."""
    undiscounted_cashflows: np.ndarray   # (n_paths, n_observations) flux bruts par date
    payment_times: np.ndarray            # (n_paths,) date effective du dernier flux (autocall ou maturité)
    autocalled: np.ndarray               # (n_paths,) bool, True si autocall déclenché
    knocked_in: np.ndarray               # (n_paths,) bool, True si barrière knock-in touchée
    total_payoff: np.ndarray             # (n_paths,) flux total non actualisé (avant discount)


def compute_phoenix_payoff(
    paths: np.ndarray,
    specs: ProductSpecs,
    spot_initial: float,
    continuous_min_ratio: np.ndarray | None = None,
) -> PayoffResult:
    """
    Calcule le payoff Phoenix Autocall pour un ensemble de trajectoires simulées.

    Parameters
    ----------
    paths : np.ndarray
        Trajectoires simulées aux dates d'observation, forme (n_paths, n_observations)
    specs : ProductSpecs
        Spécifications du produit (barrières, coupon, knock-in)
    spot_initial : float
        Prix initial du sous-jacent (S0), pour normaliser les trajectoires
    continuous_min_ratio : np.ndarray, optional
        Si le knock-in est observé en continu, tableau (n_paths,) donnant le minimum
        de S_t/S0 sur toute la durée de vie (pas seulement aux dates d'observation).
        Obligatoire si specs.knock_in_observation == "continuous".

    Returns
    -------
    PayoffResult
    """
    n_paths, n_obs = paths.shape
    if n_obs != specs.n_observations:
        raise ValueError("Le nombre de colonnes de `paths` doit correspondre à specs.n_observations.")

    ratios = paths / spot_initial  # s_i = S_i / S0, forme (n_paths, n_obs)

    # --- 1. Détection de l'autocall (première date où barrière franchie) ---
    autocall_triggered = ratios >= specs.autocall_barriers[np.newaxis, :]  # (n_paths, n_obs) bool

    has_autocalled = autocall_triggered.any(axis=1)
    # indice de la première date d'autocall (n_obs si jamais déclenché, cf. argmax sur bool)
    first_autocall_idx = np.where(
        has_autocalled,
        autocall_triggered.argmax(axis=1),
        n_obs,  # sentinel : pas d'autocall
    )

    # --- 2. Détection du respect de la barrière de coupon à chaque date ---
    coupon_eligible = ratios >= specs.coupon_barriers[np.newaxis, :]  # (n_paths, n_obs) bool

    # Une date ne "compte" pour le coupon que si le produit est encore en vie à cette date,
    # c'est-à-dire i <= indice d'autocall (le coupon est aussi versé le jour de l'autocall)
    obs_indices = np.arange(n_obs)[np.newaxis, :]  # (1, n_obs)
    alive_mask = obs_indices <= first_autocall_idx[:, np.newaxis]  # (n_paths, n_obs) bool

    coupon_eligible_alive = coupon_eligible & alive_mask

    # --- 3. Effet mémoire : à chaque date éligible, on rattrape les coupons manqués ---
    cashflows = np.zeros((n_paths, n_obs))

    if specs.memory_effect:
        # Pour chaque trajectoire, on compte le nombre de périodes depuis le dernier coupon versé.
        # Astuce vectorisée : on numérote les dates (1..n_obs), on prend le "dernier index éligible
        # précédent" via un cummax des indices éligibles, puis on déduit le nombre de périodes couvertes.
        period_index = np.arange(1, n_obs + 1)[np.newaxis, :]  # (1, n_obs), 1-indexed

        # dernier index (1-indexed) où un coupon a été versé, avant ou à la date courante (0 si aucun)
        eligible_period_index = np.where(coupon_eligible_alive, period_index, 0)
        last_paid_index = np.maximum.accumulate(eligible_period_index, axis=1)
        # décaler d'une colonne vers la droite pour avoir "le dernier versement AVANT la date courante"
        last_paid_before = np.concatenate(
            [np.zeros((n_paths, 1)), last_paid_index[:, :-1]], axis=1
        )

        periods_covered = np.where(
            coupon_eligible_alive,
            period_index - last_paid_before,
            0,
        )
        cashflows += periods_covered * specs.coupon_rate_period * specs.nominal
    else:
        # Sans effet mémoire : un coupon simple à chaque date éligible, pas de rattrapage
        cashflows += coupon_eligible_alive * specs.coupon_rate_period * specs.nominal

    # --- 4. Détermination du knock-in ---
    if specs.knock_in_observation == "continuous":
        if continuous_min_ratio is None:
            raise ValueError(
                "continuous_min_ratio est requis quand knock_in_observation == 'continuous'. "
                "TODO: fournir les trajectoires fines via simulate_gbm_full_paths."
            )
        knocked_in = continuous_min_ratio < specs.knock_in_barrier
        final_ratio = ratios[:, -1]
    else:  # at_maturity
        final_ratio = ratios[:, -1]
        knocked_in = final_ratio < specs.knock_in_barrier

    # --- 5. Remboursement du capital ---
    capital_repayment = np.zeros(n_paths)

    # Cas autocall : capital remboursé à 100% à la date d'autocall
    capital_repayment = np.where(has_autocalled, specs.nominal, capital_repayment)

    # Cas maturité sans autocall : dépend du knock-in
    not_autocalled = ~has_autocalled
    capital_repayment = np.where(
        not_autocalled & ~knocked_in,
        specs.nominal,
        capital_repayment,
    )
    capital_repayment = np.where(
        not_autocalled & knocked_in,
        specs.nominal * final_ratio,
        capital_repayment,
    )

    # Le capital est versé à la date de sortie (autocall ou maturité)
    exit_idx = np.minimum(first_autocall_idx, n_obs - 1)
    cashflows[np.arange(n_paths), exit_idx] += capital_repayment

    # Annuler tous les flux de coupon postérieurs à la sortie (sécurité, déjà géré par alive_mask
    # mais on s'assure qu'aucun flux ne traîne après l'indice de sortie)
    post_exit_mask = obs_indices > exit_idx[:, np.newaxis]
    cashflows = np.where(post_exit_mask, 0.0, cashflows)

    payment_times = specs.observation_times[exit_idx]
    total_payoff = cashflows.sum(axis=1)

    return PayoffResult(
        undiscounted_cashflows=cashflows,
        payment_times=payment_times,
        autocalled=has_autocalled,
        knocked_in=knocked_in,
        total_payoff=total_payoff,
    )
