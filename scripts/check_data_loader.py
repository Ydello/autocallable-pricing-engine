"""
check_data_loader.py
=====================
Script MANUEL (pas un test pytest) pour vérifier que data_loader.py arrive
bien à se connecter à yfinance et FRED avec de vraies données, en local.

Usage :
    python scripts/check_data_loader.py

Ce script n'est PAS lancé par pytest (il fait de vrais appels réseau, ce qui
serait lent et fragile dans une suite de tests automatisée). Il sert juste à
vérifier "à la main" que tout fonctionne chez toi.
"""

import sys
import os
import traceback

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from data_loader import (
    fetch_underlying_history,
    fetch_risk_free_rate,
    fetch_volatility_proxy,
    build_market_data_bundle,
)

START_DATE = "2023-01-01"


def check(label, func):
    print(f"--- {label} ---")
    try:
        result = func()
        print("OK")
        print(result)
    except Exception as e:
        print("ECHEC:", e)
        traceback.print_exc()
    print()


def main():
    check(
        "1. fetch_underlying_history('^GSPC')",
        lambda: fetch_underlying_history("^GSPC", start=START_DATE).tail(),
    )

    check(
        "2. fetch_volatility_proxy('^VIX')",
        lambda: fetch_volatility_proxy("^VIX", start=START_DATE),
    )

    check(
        "3. fetch_risk_free_rate('DGS3MO')",
        lambda: fetch_risk_free_rate("DGS3MO", start=START_DATE),
    )

    print("--- 4. build_market_data_bundle (config réel) ---")
    try:
        import yaml
        config_path = os.path.join(os.path.dirname(__file__), "..", "config", "product_config.yaml")
        with open(config_path) as f:
            config = yaml.safe_load(f)

        bundle = build_market_data_bundle(config)
        print("OK")
        print("Spot          :", bundle.spot_price)
        print("Pricing date  :", bundle.pricing_date)
        print("Taux sans risque :", bundle.risk_free_rate)
        print("Vol implicite (VIX) :", bundle.implied_vol_proxy)
        print("Dividend yield :", bundle.dividend_yield, "(None attendu à ce stade)")
    except Exception as e:
        print("ECHEC:", e)
        traceback.print_exc()


if __name__ == "__main__":
    main()
