"""Check the compiled ML objective against the independent NumPy path."""

from copy import deepcopy
import pickle

import jax
import numpy as np
import pytest

from bvar.models import NaturalConjugate
from bvar.models.conjugate import jax_marginal_likelihood as jax_ml
from bvar.models.conjugate import marginal_likelihood as ml
from bvar.utils import construct_Y_Z


def _inputs(
    *,
    soc=True,
    sur=True,
    minnesota=True,
    covid=False,
    levels=(True, False),
    custom=False,
    add_priors=True,
    effective=None,
    seed=83,
):
    """Reuse two-variable, two-lag shapes to keep compilation costs small."""
    rng = np.random.default_rng(seed)
    data = rng.normal(0, 0.3, (48, 2))
    data[:, 0] += 1.5 + np.linspace(0, 0.6, len(data))
    data[:, 1] += 0.7
    indices = np.array([25, 26], dtype=int) if covid else np.array([], dtype=int)
    model = NaturalConjugate(
        minnesota=minnesota,
        soc=soc,
        sur=sur,
        covid=covid,
    )
    model.pars.nu_0 = 7
    model.pars.S_0 = np.diag([0.8, 1.3])
    # A finite intercept variance keeps dummy-only precision well conditioned.
    model.pars.lambda_constant = 3.0
    if custom:
        model.pars.nu_0 = 9.5
        model.pars.S_0 = np.diag([1.1, 0.6])
        model.pars.lambda_constant = 1.7
        model.pars.lambda_covid = 2.3
        for name, shape, scale in (
            ("c1", 2.4, 0.35),
            ("c3", 3.1, 0.8),
            ("mu", 1.7, 1.2),
            ("theta", 4.2, 0.6),
        ):
            setattr(model.pars, name + "_k", shape)
            setattr(model.pars, name + "_theta", scale)
    Y, Z = construct_Y_Z(data, 2, indices)
    return dict(
        data=data,
        n_lags=2,
        covid_indices=indices,
        levels=np.array(levels),
        model=model,
        Y=Y,
        Z=Z,
        add_priors=add_priors,
        soc=None if effective is None else effective[0],
        sur=None if effective is None else effective[1],
    )


def _vectors(model):
    for c1, c3, mu, theta in ((0.3, 1.6, 0.8, 1.3), (0.8, 2.4, 1.7, 0.6)):
        values = [c1, c3] + ([mu] if model.soc else [])
        values += [theta] if model.sur else []
        yield np.log(np.expm1(values))


def _numpy_value(x, inputs):
    # The reference fills model parameters in place; isolate every evaluation.
    args = dict(inputs, model=deepcopy(inputs["model"]))
    for name in ("soc", "sur"):
        if args[name] is None:
            args[name] = getattr(args["model"], name)
    return ml.objective_function(x, nb_dummy_obs=0, **args)


def _central_gradient(x, inputs):
    # eps**(1/3) balances O(h**2) truncation and O(eps/h) cancellation.
    steps = np.cbrt(np.finfo(np.float64).eps) * np.maximum(1, np.abs(x))
    gradient = np.empty_like(x)
    for i, step in enumerate(steps):
        offset = np.zeros_like(x)
        offset[i] = step
        gradient[i] = (
            _numpy_value(x + offset, inputs) - _numpy_value(x - offset, inputs)
        ) / (2 * step)
    return gradient


def _assert_matches_numpy(objective, x, inputs):
    value, gradient = objective(x)
    assert isinstance(value, float)
    assert gradient.shape == x.shape
    assert gradient.dtype == np.float64
    assert np.isfinite(value)
    assert np.all(np.isfinite(gradient))
    # Small, conditioned systems should agree well below 1e-8 in value.
    np.testing.assert_allclose(
        value,
        _numpy_value(x, inputs),
        atol=2e-9,
        rtol=2e-11,
    )
    # Differencing objectives of order 100 loses several digits. Absolute
    # tolerance covers near-zero derivatives; relative tolerance is 2 ppm.
    np.testing.assert_allclose(
        gradient,
        _central_gradient(x, inputs),
        atol=2e-7,
        rtol=2e-6,
    )
    return value, gradient


@pytest.mark.parametrize(
    "options",
    [
        pytest.param(dict(soc=False, sur=False), id="no-dummies"),
        pytest.param(dict(soc=True, sur=False), id="soc-only"),
        pytest.param(dict(soc=False, sur=True), id="sur-only"),
        pytest.param(dict(), id="soc-sur-mixed-levels"),
        pytest.param(dict(levels=(True, True)), id="all-levels"),
        pytest.param(dict(levels=(False, False)), id="all-stationary"),
        pytest.param(dict(effective=(False, False)), id="both-disabled"),
        pytest.param(dict(effective=(True, False)), id="sur-disabled"),
        pytest.param(dict(effective=(False, True)), id="soc-disabled"),
        pytest.param(dict(covid=True, custom=True), id="covid-custom-priors"),
        pytest.param(dict(add_priors=False, custom=True), id="no-hyperpriors"),
        pytest.param(
            dict(add_priors=False, effective=(False, False)),
            id="disabled-without-hyperpriors",
        ),
        pytest.param(dict(minnesota=False, soc=False, sur=False), id="no-minnesota"),
        pytest.param(
            dict(minnesota=False, effective=(False, False), add_priors=False),
            id="no-minnesota-disabled-without-hyperpriors",
        ),
    ],
)
def test_fixed_values_and_gradients_match_numpy(options):
    inputs = _inputs(**options)
    objective = jax_ml.build_objective(**inputs)
    for x in _vectors(inputs["model"]):
        _, gradient = _assert_matches_numpy(objective, x, inputs)
        if not inputs["add_priors"]:
            if not inputs["model"].minnesota:
                np.testing.assert_array_equal(gradient[:2], 0)
            for i, name in enumerate(("soc", "sur"), start=2):
                if inputs[name] is False:
                    assert gradient[i] == 0


def test_build_and_evaluation_do_not_mutate_inputs():
    inputs = _inputs(covid=True, custom=True)
    before = pickle.dumps(inputs)
    objective = jax_ml.build_objective(**inputs)
    assert pickle.dumps(inputs) == before
    for x in _vectors(inputs["model"]):
        original_x = x.copy()
        first = objective(x)
        objective(x + 0.1)
        repeated = objective(x)
        np.testing.assert_array_equal(x, original_x)
        assert pickle.dumps(inputs) == before
        assert repeated[0] == first[0]
        np.testing.assert_array_equal(repeated[1], first[1])


@pytest.mark.parametrize("enabled", [False, True], ids=["x32-default", "x64-default"])
def test_double_precision_is_scoped(enabled):
    original_config = dict(jax.config.values)
    with jax_ml.enable_x64(enabled):
        assert jax.config.x64_enabled is enabled
        expected_config = dict(jax.config.values)
        inputs = _inputs()
        objective = jax_ml.build_objective(**inputs)
        assert jax.config.values == expected_config
        x = next(_vectors(inputs["model"]))
        _assert_matches_numpy(objective, x, inputs)
        assert jax.config.values == expected_config
        value, gradient = objective(np.full_like(x, np.nan))
        assert value == np.inf
        assert np.all(np.isnan(gradient))
        assert jax.config.values == expected_config
    assert jax.config.values == original_config


def test_same_shape_fits_do_not_reuse_stale_data_or_priors():
    cases = [
        _inputs(covid=True),
        _inputs(covid=True, seed=97),
        _inputs(covid=True, custom=True),
        _inputs(covid=True, levels=(False, True)),
    ]
    moved_covid = deepcopy(cases[0])
    moved_covid["covid_indices"] = np.array([31, 32])
    moved_covid["Y"], moved_covid["Z"] = construct_Y_Z(
        moved_covid["data"],
        moved_covid["n_lags"],
        moved_covid["covid_indices"],
    )
    cases.append(moved_covid)
    x = next(_vectors(cases[0]["model"]))
    first = jax_ml.build_objective(**cases[0])
    baseline = _assert_matches_numpy(first, x, cases[0])
    objectives = [first]
    for inputs in cases[1:]:
        assert inputs["Y"].shape == cases[0]["Y"].shape
        assert inputs["Z"].shape == cases[0]["Z"].shape
        objective = jax_ml.build_objective(**inputs)
        objectives.append(objective)
        value, _ = _assert_matches_numpy(objective, x, inputs)
        assert abs(value - baseline[0]) > 1e-4
        repeated = first(x)
        assert repeated[0] == baseline[0]
        np.testing.assert_array_equal(repeated[1], baseline[1])
    # Revisit each closure after every other fit has populated the JIT cache.
    for objective, inputs in zip(reversed(objectives), reversed(cases)):
        _assert_matches_numpy(objective, x, inputs)


def test_invalid_trials_are_rejected_without_poisoning_valid_evaluations():
    inputs = _inputs()
    objective = jax_ml.build_objective(**inputs)
    x = next(_vectors(inputs["model"]))
    baseline = objective(x)
    assert np.isfinite(baseline[0])
    # Underflowed hyperparameters must not turn dummy-only subtraction into an
    # apparent improvement. Huge lag decay also overflows prior precision.
    for index, bad_value in (
        (0, -1000.0),
        (1, 2000.0),
        (2, -1000.0),
        (3, -1000.0),
        (0, np.nan),
        (0, np.inf),
        (0, -np.inf),
    ):
        trial = x.copy()
        trial[index] = bad_value
        value, gradient = objective(trial)
        assert value == np.inf, (index, bad_value, value)
        assert value > baseline[0]
        assert gradient.shape == x.shape
        assert np.all(np.isnan(gradient))
        repeated = objective(x)
        assert repeated[0] == baseline[0]
        np.testing.assert_array_equal(repeated[1], baseline[1])


def test_nonfinite_gradient_rejects_an_apparently_improved_value(monkeypatch):
    inputs = _inputs()
    objective = jax_ml.build_objective(**inputs)
    x = next(_vectors(inputs["model"]))
    baseline, _ = objective(x)
    for invalid in (np.nan, np.inf, -np.inf):
        bad_gradient = np.zeros_like(x)
        bad_gradient[2] = invalid
        monkeypatch.setattr(
            jax_ml,
            "_value_and_grad",
            lambda x, inputs: (baseline - 1e6, bad_gradient),
        )
        value, gradient = objective(x)
        assert value == np.inf
        assert np.all(np.isnan(gradient))
