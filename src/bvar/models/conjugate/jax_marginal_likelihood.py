"""Compiled, double-precision ML objective and gradient for SciPy BFGS.

Only hyperparameters are differentiated. Data and prior templates are dynamic
JIT inputs, so fits with the same shapes reuse compiled code without retaining
an earlier fit's data. NumPy builds the templates once, outside the hot loop.
"""

from copy import deepcopy

import jax
import jax.numpy as jnp
from jax.scipy.linalg import cho_solve
import numpy as np
from scipy.special import gammaln

try:
    from jax import enable_x64
except ImportError:  # JAX versions supporting Python 3.10.
    from jax.experimental import enable_x64

from ...dummy_observations import stack_dummies
from .minnesota import prior_minnesota


def _logml(Y, Z, mean, precision, psi, nu, constant):
    """Residual-based conjugate integral, with diagonal prior precision."""
    n = Y.shape[1]
    factor = jnp.linalg.cholesky(Z.T @ Z + jnp.diag(precision))
    estimate = cho_solve((factor, True), Z.T @ Y + precision[:, None] * mean)
    logdet_a = 2 * jnp.log(jnp.diag(factor)).sum() - jnp.log(precision).sum()
    residual = Y - Z @ estimate
    deviation = estimate - mean
    middle = residual.T @ residual + deviation.T @ (precision[:, None] * deviation)
    inv_scale = 1 / jnp.sqrt(psi)
    scale = jnp.eye(n) + inv_scale[:, None] * middle * inv_scale[None, :]
    scale_factor = jnp.linalg.cholesky(scale)
    logdet_b = 2 * jnp.log(jnp.diag(scale_factor)).sum()
    return constant - n * logdet_a / 2 - (len(Y) + nu) * logdet_b / 2


def _objective(x, inputs):
    hp = jax.nn.softplus(x)
    log_hp = jnp.log(hp)
    precision = inputs["precision"] * jnp.exp(
        inputs["log_lags"] * hp[1] - 2 * inputs["shrink"] * log_hp[0]
    )
    scales = jnp.exp(-inputs["dummy_weights"] @ log_hp)
    yd = inputs["yd"] * scales[:, None]
    zd = inputs["zd"] * scales[:, None]
    Y = jnp.concatenate((yd, inputs["Y"]))
    Z = jnp.concatenate((zd, inputs["Z"]))
    prior = (inputs["mean"], precision, inputs["psi"], inputs["nu"])
    value = _logml(Y, Z, *prior, inputs["constant"])
    if yd.shape[0]:
        value -= _logml(yd, zd, *prior, inputs["dummy_constant"])
    value += jnp.sum(inputs["gamma_log"] * log_hp - inputs["gamma_rate"] * hp)
    value += inputs["gamma_constant"]
    return -value


_value_and_grad = jax.jit(jax.value_and_grad(_objective))


def build_objective(
    data,
    n_lags,
    covid_indices,
    levels,
    model,
    Y,
    Z,
    add_priors=True,
    soc=None,
    sur=None,
):
    """Prepare a pure ``(value, gradient)`` callable for ``minimize(jac=True)``.

    The model is not mutated. Double precision is scoped to this callable,
    leaving the application's JAX configuration unchanged. Non-finite values
    or gradients reject a trial point with infinity rather than letting a
    failed dummy-only factorisation look like an improved likelihood.
    """
    soc = model.soc if soc is None else soc
    sur = model.sur if sur is None else sur
    template = deepcopy(model)
    template.pars.c1 = 1.0
    template.pars.c3 = 0.0
    n, k = Y.shape[1], Z.shape[1]
    shrink, log_lags = np.zeros(k), np.zeros(k)
    if model.minnesota:
        beta, precision_matrix = prior_minnesota(
            data,
            n_lags,
            covid_indices,
            levels,
            template.pars,
        )
        precision = np.diag(precision_matrix)
        shrink[1 : 1 + n * n_lags] = 1
        log_lags[1 : 1 + n * n_lags] = np.repeat(np.log(np.arange(1, n_lags + 1)), n)
    else:
        beta, precision = np.zeros(n * k), np.full(k, 1e-10)

    # Layout follows model flags; effective flags only decide which rows exist.
    names = (
        ["c1", "c3"] + (["mu"] if model.soc else []) + (["theta"] if model.sur else [])
    )
    for name in names[2:]:
        setattr(template.pars, name, 1.0)
    Ya, Za, nd = stack_dummies(
        Y,
        Z,
        n_lags,
        levels,
        template,
        covid_indices,
        soc=soc,
        sur=sur,
    )
    weights = np.zeros((nd, len(names)))
    if sur and model.sur:
        weights[0, names.index("theta")] = 1
    if soc and model.soc:
        weights[int(sur) :, names.index("mu")] = 1

    psi, nu = np.diag(model.pars.S_0), model.pars.nu_0

    def constant(rows):
        return (
            -n * rows * np.log(np.pi) / 2
            - rows * np.log(psi).sum() / 2
            + np.sum(
                gammaln((rows + nu - np.arange(n)) / 2)
                - gammaln((nu - np.arange(n)) / 2)
            )
        )

    shape = np.array([getattr(model.pars, name + "_k") for name in names])
    scale = np.array([getattr(model.pars, name + "_theta") for name in names])
    inputs = {
        "Y": Y,
        "Z": Z,
        "yd": Ya[:nd],
        "zd": Za[:nd],
        "mean": beta.reshape(n, k).T,
        "precision": precision,
        "shrink": shrink,
        "log_lags": log_lags,
        "dummy_weights": weights,
        "psi": psi,
        "nu": nu,
        "constant": constant(len(Ya)),
        "dummy_constant": constant(nd),
        "gamma_log": shape - 1 if add_priors else np.zeros_like(shape),
        "gamma_rate": 1 / scale if add_priors else np.zeros_like(scale),
        "gamma_constant": -np.sum(shape * np.log(scale) + gammaln(shape))
        if add_priors
        else 0.0,
    }
    with enable_x64():
        inputs = jax.tree.map(lambda a: jnp.asarray(a, dtype=jnp.float64), inputs)

    def value_and_grad(x):
        with enable_x64():
            value, gradient = _value_and_grad(jnp.asarray(x, dtype=jnp.float64), inputs)
            value, gradient = float(value), np.asarray(gradient, dtype=np.float64)
        if not np.isfinite(value) or not np.all(np.isfinite(gradient)):
            return np.inf, np.full_like(gradient, np.nan)
        return value, gradient

    return value_and_grad
