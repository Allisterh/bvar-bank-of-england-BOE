"""Numerical guards for the ML speed edits; the old kernel lives only in tests."""

from itertools import product
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from scipy.special import gammaln

from bvar import BVAR
from bvar.models import NaturalConjugate
from bvar.models.conjugate import marginal_likelihood as ml
from bvar.utils import construct_Y_Z


def reference_logml(Y, Z, beta_0, V_A_inv, S_0, nu_0, n):
    """Frozen pre-optimisation eigenvalue formulation (77e2976)."""
    b = beta_0.reshape(n, -1).T
    T = len(Y)
    psi = np.diag(S_0)
    try:
        estimate = np.linalg.solve(Z.T @ Z + V_A_inv, Z.T @ Y + V_A_inv @ b)
    except np.linalg.LinAlgError:
        estimate = b
    residuals = Y - Z @ estimate
    sqrt_omega = np.diag(1 / np.sqrt(np.diag(V_A_inv)))
    inv_sqrt_psi = np.diag(1 / np.sqrt(psi))
    middle = residuals.T @ residuals + (estimate - b).T @ V_A_inv @ (estimate - b)
    eig_a = np.real(np.linalg.eigvals(sqrt_omega @ (Z.T @ Z) @ sqrt_omega))
    eig_b = np.real(np.linalg.eigvals(inv_sqrt_psi @ middle @ inv_sqrt_psi))
    eig_a[eig_a < 1e-12] = 0
    eig_b[eig_b < 1e-12] = 0
    value = (
        -n * T * np.log(np.pi) / 2
        + np.sum(gammaln((T + nu_0 - np.arange(n)) / 2)
                 - gammaln((nu_0 - np.arange(n)) / 2))
        - T * np.log(psi).sum() / 2
        - n * np.log1p(eig_a).sum() / 2
        - (T + nu_0) * np.log1p(eig_b).sum() / 2
    )
    return value if np.isfinite(value) else -1e5


@pytest.mark.parametrize("soc,sur", list(product([False, True], repeat=2)))
@pytest.mark.parametrize("levels", [False, True])
@pytest.mark.parametrize("covid", [False, True])
@pytest.mark.parametrize("lags", [1, 5])
@pytest.mark.parametrize("hyperparameters", [
    [0.2, 2, 1, 1], [0.01, 4, 0.02, 0.02], [5, 0.1, 20, 20],
])
def test_full_objective_matches_original(
    monkeypatch, soc, sur, levels, covid, lags, hyperparameters,
):
    """Include dummy-only subtraction and Gamma/softplus terms at fixed vectors.

    Absolute tolerance covers rounding in small objectives; relative tolerance
    permits at most 1e-8 of large log likelihoods with diffuse priors.
    """
    rng = np.random.default_rng(83)
    data = rng.normal(0, 0.05, (90, 4))
    if levels:
        data = 4 + np.cumsum(data + 0.002, axis=0)
    indices = np.arange(50, 54) if covid else np.array([], dtype=int)
    level_flags = np.full(4, levels)
    model = NaturalConjugate(soc=soc, sur=sur, covid=covid)
    model.pars.nu_0 = 8
    model.pars.S_0 = np.diag([0.01, 0.02, 0.03, 0.04])
    Y, Z = construct_Y_Z(data, lags, indices)
    values = hyperparameters[:2] + ([hyperparameters[2]] if soc else [])
    values += [hyperparameters[3]] if sur else []
    pars = np.log(np.expm1(values))
    args = (pars, data, lags, indices, level_flags, 0, model, Y, Z, True, soc, sur)
    actual = ml.objective_function(*args)
    with monkeypatch.context() as context:
        context.setattr(ml, "log_marginal_likelihood", reference_logml)
        expected = ml.objective_function(*args)
    np.testing.assert_allclose(actual, expected, atol=1e-6, rtol=1e-8)


@pytest.mark.parametrize("rows", [0, 1, 3, 80])
@pytest.mark.parametrize("precision", [1e-6, 1.0, 1e6])
def test_rank_deficient_design_matches_original(rows, precision):
    """Dummy-only samples can have fewer rows than regressors, including zero."""
    rng = np.random.default_rng(37)
    Z = rng.normal(size=(rows, 8))
    Z[:, -1] = Z[:, 0] + 1e-10 * Z[:, 1]
    Y = rng.normal(size=(rows, 3))
    args = (Y, Z, np.zeros(24), np.eye(8) * precision, np.eye(3), 7, 3)
    np.testing.assert_allclose(
        ml.log_marginal_likelihood(*args), reference_logml(*args),
        atol=1e-6, rtol=1e-8,
    )


def test_logml_does_not_mutate_inputs():
    rng = np.random.default_rng(42)
    arrays = [
        rng.normal(size=(30, 3)), rng.normal(size=(30, 7)),
        rng.normal(size=21), np.eye(7), np.eye(3),
    ]
    copies = [a.copy() for a in arrays]
    ml.log_marginal_likelihood(*arrays, 7, 3)
    for actual, expected in zip(arrays, copies):
        np.testing.assert_array_equal(actual, expected)


@pytest.mark.parametrize("target,bad_value", [
    *product(["precision", "scale"], [0.0, -1.0, np.nan, np.inf]),
    ("data", np.nan), ("data", np.inf),
])
def test_logml_rejects_invalid_inputs(target, bad_value):
    Y, Z = np.ones((12, 2)), np.ones((12, 3))
    precision, scale = np.eye(3), np.eye(2)
    if target == "data":
        Y[0, 0] = bad_value
    elif target == "precision":
        precision[0, 0] = bad_value
    else:
        scale[0, 0] = bad_value
    with np.errstate(invalid="ignore", divide="ignore"):
        assert ml.log_marginal_likelihood(Y, Z, np.zeros(6), precision, scale, 6, 2) == -1e5


@pytest.mark.parametrize("failure_call", [1, 2])
def test_logml_rejects_failed_cholesky(monkeypatch, failure_call):
    original = ml.cho_factor
    calls = 0

    def fail_factor(*args, **kwargs):
        nonlocal calls
        calls += 1
        if calls == failure_call:
            raise np.linalg.LinAlgError("not positive definite")
        return original(*args, **kwargs)

    monkeypatch.setattr(ml, "cho_factor", fail_factor)
    value = ml.log_marginal_likelihood(
        np.ones((12, 2)), np.ones((12, 3)), np.zeros(6), np.eye(3), np.eye(2), 6, 2,
    )
    assert value == -1e5
    assert calls == failure_call


def test_logml_reuses_precision_factor(monkeypatch):
    original_factor = ml.cho_factor
    original_solve = ml.cho_solve
    factors = []

    def record_factor(*args, **kwargs):
        factor = original_factor(*args, **kwargs)
        factors.append(factor)
        return factor

    def check_solve(factor, *args, **kwargs):
        assert factor is factors[0]
        return original_solve(factor, *args, **kwargs)

    monkeypatch.setattr(ml, "cho_factor", record_factor)
    monkeypatch.setattr(ml, "cho_solve", check_solve)
    value = ml.log_marginal_likelihood(
        np.ones((12, 2)), np.ones((12, 3)), np.zeros(6), np.eye(3), np.eye(2), 6, 2,
    )
    assert np.isfinite(value) and value != -1e5
    assert len(factors) == 2


@pytest.mark.parametrize("name,length,covid", [
    ("pre_covid", 160, False), ("covid", 168, True), ("recent", 184, True),
])
def test_benchmark_fixed_and_fitted_objectives(name, length, covid):
    """Check captured original values, including its selected BFGS solution.

    Fixed probes use 1e-6 absolute tolerance. The fitted COVID vector has a
    dummy precision condition number near 6e12: 50-digit decimal algebra
    measured objective errors of 9.9e-7 (old) and 3.7e-7 (new). Allow 2e-6
    there rather than treating the noisy original fitted value as exact.
    """
    path = Path(__file__).resolve().parents[1] / "benchmarks/ml-speed/baseline.json"
    baseline = json.loads(path.read_text())["cases"][name]
    rng = np.random.default_rng(1234)
    innovations = rng.normal(0, 0.02, (184, 19))
    innovations += rng.normal(0, 0.01, (184, 1))
    innovations[160:168] *= 4
    values = 4 + np.cumsum(0.002 + innovations, axis=0)
    data = pd.DataFrame(
        values[:length], index=pd.period_range("1980Q1", periods=length, freq="Q"),
    )
    fit = BVAR(5, NaturalConjugate(soc=True, sur=True, covid=covid), stationary=False)
    array = fit._validate_and_prepare_data(data)
    model = fit.model
    model.pars.S_0, model.pars.nu_0 = model._compute_S0_nu0(array, 19, fit.covid_indices)
    Y, Z = construct_Y_Z(array, 5, fit.covid_indices)
    vectors = baseline["fixed_vectors"] + [baseline["optimum"]["x"]]
    expected = baseline["fixed_objectives"] + [baseline["optimum"]["fun"]]
    actual = [ml.objective_function(
        np.array(x), array, 5, fit.covid_indices, fit.vars_in_levels, 0,
        model, Y, Z, True, True, True,
    ) for x in vectors]
    np.testing.assert_allclose(actual[:-1], expected[:-1], atol=1e-6, rtol=0)
    np.testing.assert_allclose(actual[-1], expected[-1], atol=2e-6, rtol=0)