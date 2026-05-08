"""Tests for cdans.ci_tests."""

import numpy as np
import pytest

from cdans.ci_tests import FisherZ, get_ci_test


def test_fisher_z_independent_data():
    rng = np.random.default_rng(0)
    x = rng.standard_normal(500)
    y = rng.standard_normal(500)
    p = FisherZ().pvalue(x, y)
    assert p > 0.01, f"independent data should not be flagged as dependent (p={p})"


def test_fisher_z_dependent_data():
    rng = np.random.default_rng(0)
    x = rng.standard_normal(500)
    noise = rng.standard_normal(500) * 0.3
    y = 0.8 * x + noise
    p = FisherZ().pvalue(x, y)
    assert p < 0.001, f"dependent data should be flagged (p={p})"


def test_fisher_z_conditional_independence():
    """X and Y are conditionally independent given Z when Y = f(Z) + noise."""
    rng = np.random.default_rng(0)
    z = rng.standard_normal(500)
    x = z + rng.standard_normal(500) * 0.5
    y = z + rng.standard_normal(500) * 0.5
    # Marginally x and y look correlated (both depend on z)
    p_marginal = FisherZ().pvalue(x, y)
    assert p_marginal < 0.01
    # But conditionally on z they are independent
    p_cond = FisherZ().pvalue(x, y, z.reshape(-1, 1))
    assert p_cond > 0.05, f"should be conditionally independent (p={p_cond})"


def test_get_ci_test_factory():
    test = get_ci_test("fisherz")
    assert isinstance(test, FisherZ)
    assert test.name == "fisherz"


def test_get_ci_test_passthrough():
    fz = FisherZ()
    assert get_ci_test(fz) is fz


def test_get_ci_test_unknown():
    with pytest.raises(ValueError, match="unknown CI test"):
        get_ci_test("nonsense_test")
