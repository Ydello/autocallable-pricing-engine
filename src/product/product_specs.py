"""
product_specs.py
=================
Représente les caractéristiques contractuelles du produit Phoenix Autocall,
chargées depuis config/product_config.yaml.

TODO (à finaliser ensemble) :
- Valider la génération automatique des dates d'observation selon la fréquence
- Finaliser les valeurs numériques des barrières (step-down, coupon, knock-in)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import yaml


@dataclass
class ProductSpecs:
    nominal: float
    maturity_years: float
    observation_times: np.ndarray          # en années, ex. [0.25, 0.5, ..., 3.0]
    autocall_barriers: np.ndarray           # une valeur par date d'observation (fraction du spot initial)
    coupon_barriers: np.ndarray             # idem
    coupon_rate_period: float               # taux de coupon par période (pas annualisé)
    memory_effect: bool
    knock_in_barrier: float
    knock_in_observation: str               # "at_maturity" ou "continuous"

    @classmethod
    def from_config(cls, config_path: str) -> "ProductSpecs":
        """Construit un ProductSpecs à partir du fichier YAML de configuration."""
        with open(config_path, "r", encoding="utf-8") as f:
            cfg = yaml.safe_load(f)

        product_cfg = cfg["product"]
        n_obs = product_cfg["n_observations"]
        maturity = product_cfg["maturity_years"]

        observation_times = np.linspace(maturity / n_obs, maturity, n_obs)

        autocall_cfg = product_cfg["autocall_barrier"]
        if autocall_cfg.get("values") is not None:
            autocall_barriers = np.array(autocall_cfg["values"], dtype=float)
        elif autocall_cfg["type"] == "step_down":
            initial = autocall_cfg["initial_level"]
            step = autocall_cfg["step"]
            autocall_barriers = np.array([initial - step * i for i in range(n_obs)])
        else:  # flat
            autocall_barriers = np.full(n_obs, autocall_cfg["initial_level"])

        coupon_cfg = product_cfg["coupon_barrier"]
        coupon_barriers = np.full(n_obs, coupon_cfg["level"])

        coupon_rate_annual = product_cfg["coupon"]["rate_annual"]
        periods_per_year = n_obs / maturity
        coupon_rate_period = coupon_rate_annual / periods_per_year

        return cls(
            nominal=product_cfg["nominal"],
            maturity_years=maturity,
            observation_times=observation_times,
            autocall_barriers=autocall_barriers,
            coupon_barriers=coupon_barriers,
            coupon_rate_period=coupon_rate_period,
            memory_effect=product_cfg["coupon"]["memory_effect"],
            knock_in_barrier=product_cfg["capital_protection"]["knock_in_barrier"],
            knock_in_observation=product_cfg["capital_protection"]["observation_type"],
        )

    @property
    def n_observations(self) -> int:
        return len(self.observation_times)

    def validate(self) -> None:
        """
        Vérifie la cohérence interne des specs.

        TODO : étoffer avec d'autres règles métier au fur et à mesure
               (ex. coupon_barrier <= autocall_barrier à chaque date)
        """
        if len(self.autocall_barriers) != self.n_observations:
            raise ValueError("autocall_barriers doit avoir une valeur par date d'observation.")
        if len(self.coupon_barriers) != self.n_observations:
            raise ValueError("coupon_barriers doit avoir une valeur par date d'observation.")
        if np.any(self.coupon_barriers > self.autocall_barriers):
            raise ValueError("La barrière de coupon ne devrait pas dépasser la barrière d'autocall.")
        if self.knock_in_observation not in ("at_maturity", "continuous"):
            raise ValueError("knock_in_observation doit être 'at_maturity' ou 'continuous'.")
