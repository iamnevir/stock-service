"""
evolution_tool.py
================
High-level API for Alpha Evolution functions. 
Provides standalone functions that return dictionaries instead of saving files.
"""

import sys
import os

# Ensure the parent directory is in path for imports
sys.path.insert(0, "/home/ubuntu/nevir")

from huy.evolution.alpha_loop import (
    fetch_alphas, 
    fix_error, 
    diagnose, 
    evolve, 
    run_loop,
    combine,
    DEFAULT_MODEL
)

def get_alpha_fixes(ids: list[str], model: str = DEFAULT_MODEL) -> dict:
    """
    Stage: Fix technical errors (timeouts, logic crashes).
    Input: List of IDs/names.
    Output: Dict {alpha_name: fixed_code}.
    """
    return fix_error(ids, model=model, verbose=False)


def get_alpha_diagnosis(ids: list[str], model: str = DEFAULT_MODEL, intent: str = "flaw") -> dict:
    """
    Stage 1: Diagnose alphas.
    Input: List of IDs/names.
    Output: Diagnosis dict (diagnoses, cross_alpha_patterns).
    """
    diagnosis, _ = diagnose(ids, model=model, verbose=False, intent=intent)
    return diagnosis


def get_alpha_evolution(ids: list[str], diagnosis: dict, model: str = DEFAULT_MODEL) -> dict:
    """
    Stage 2: Evolve alphas based on a diagnosis.
    Input: List of IDs/names and a diagnosis dictionary.
    Output: Dict {variant_name: evolved_code}.
    """
    alphas = fetch_alphas(ids)
    return evolve(alphas, diagnosis, model=model, verbose=False)


def get_alpha_combination(ids: list[str], model: str = DEFAULT_MODEL) -> dict:
    """
    Stage: Combine 2-5 alphas into a single super alpha.
    Input: List of IDs/names.
    Output: Dict {combined_alpha_name: combined_code}.
    """
    return combine(ids, model=model, verbose=False)


def get_alpha_enhancement(ids: list[str], model: str = DEFAULT_MODEL) -> dict:
    """
    High-level Enhancement Pipeline: Fetch -> Fix -> Diagnose (intent='enhance') -> Evolve.
    Targeted at improving average-performing alphas with advanced optimizations.
    Input: List of IDs/names.
    Output: Dict { "diagnosis": ..., "evolved": ... }.
    """
    return run_alpha_evolution_pipeline(ids, model=model, intent="enhance")


def run_alpha_evolution_pipeline(ids: list[str], model: str = DEFAULT_MODEL, intent: str = "flaw") -> dict:
    """
    Full Pipeline: Fetch -> Fix -> Diagnose -> Evolve.
    Input: List of IDs/names.
    Output: Dict { "diagnosis": ..., "evolved": ... }.
    """
    diagnosis, evolved = run_loop(
        ids, 
        model=model, 
        save=False, 
        verbose=False, 
        intent=intent
    )
    return {
        "diagnosis": diagnosis,
        "evolved": evolved
    }

# # Example Usage:
# if __name__ == "__main__":
#     ids = ["69e9eecfc5f0365178b24a52","69e9eef9c5f0365178b24a56","69e9ef2ec5f0365178b24a58","69e9ef55c5f0365178b24a59","69e9ef99c5f0365178b24a5a"]
#     enhancements = get_alpha_enhancement(ids)
#     print(enhancements)
