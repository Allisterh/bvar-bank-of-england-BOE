"""End-to-end fixtures captured with the ML speed edits stashed.

Baseline: 77e29764b14b58dd4d5da7741d2bc65818a40738, NumPy 2.5.2,
SciPy 1.18.1, one BLAS thread. Both BFGS runs converged for each fixture.
Expected values come from the original optimiser, not the candidate kernel.
"""

import numpy as np
import pandas as pd
import pytest
from threadpoolctl import threadpool_limits

from bvar import BVAR
from bvar.models import NaturalConjugate
from bvar.models.conjugate import marginal_likelihood as ml
from bvar.utils import construct_Y_Z, simulate_var


BASELINES = {
    "stationary": {
        "objective": 238.2763148890541,
        "negative_logml": 238.27013958064023,
        "hyperparameters": [0.3358682411820485, 2.150746795644816],
        "beta": [
            2.720912547912424, 0.45192949110497305, 0.20138493308163993,
            0.05053422206880838, -0.0902270668709929, 0.9732947399813886,
            -0.053744326983841066, 0.559295693166079,
            -0.054807457986350845, 0.025524284064684252,
        ],
        "sigma": [
            1.1186717127030916, 0.41305716876501475,
            0.41305716876501614, 1.010256882137974,
        ],
        "forecast": [
            [5.158359662898083, 0.2712892609316554],
            [5.377186799902471, 0.5722064795946639],
            [5.5024516639989915, 0.7285419576501998],
        ],
    },
    "levels_soc_sur": {
        "objective": 52.98996912819578,
        "negative_logml": 50.942093133887425,
        "hyperparameters": [
            0.06281426259260678, 1.9594446504402043,
            1.0350187401101865, 0.4385667179401945,
        ],
        "beta": [
            -0.1483982951116514, 0.9752554675452753, 0.05625830955856262,
            -0.009326902366856018, 0.0285636024087345, 0.14121802821084511,
            0.007288021023092838, 0.9726731981315031,
            -0.003967466755852514, -0.003980916070733041,
        ],
        "sigma": [
            0.07770002321736802, 0.00904698748256091,
            0.009046987482559558, 0.1064049025405839,
        ],
        "forecast": [
            [5.262654727783047, 4.792449110051672],
            [5.341939456442404, 4.80147091952318],
            [5.419285348513713, 4.810454524941122],
        ],
    },
}


@pytest.mark.parametrize("case", BASELINES)
def test_entire_ml_optimisation_matches_baseline(case, monkeypatch):
    """Run real BFGS including a restart; check its selected fit end to end.

    Tolerances were fixed against the stashed baseline before testing the edits.
    The penalised optimum is locally flat, so use 1e-7 for its objective but
    1e-4 for fitted hyperparameters and the unpenalised likelihood (whose
    gradient need not vanish at that optimum). Output tolerances are 1e-5 for
    coefficients/forecasts and 1e-6 for covariance, all absolute (rtol=0).
    Neither the iteration count nor the finite-difference path must be identical.
    """
    expected = BASELINES[case]
    levels = case == "levels_soc_sur"
    with threadpool_limits(limits=1, user_api="blas"):
        if levels:
            rng = np.random.default_rng(1234)
            data = pd.DataFrame(
                np.cumsum(rng.normal(0.02, 0.3, (80, 2)), axis=0),
                index=pd.period_range("1980Q1", periods=80, freq="Q"),
            )
        else:
            data, _, _, _ = simulate_var(
                80, 2, 2, covid=False, levels=False, seed=1234,
            )
        fit = BVAR(
            2, NaturalConjugate(minnesota=True, soc=levels, sur=levels, covid=False),
            stationary=not levels, optimisation_method="ml",
        )
        results = []
        minimise = ml.minimize

        def record_result(*args, **kwargs):
            # Observe the real SciPy optimiser; do not substitute its output.
            result = minimise(*args, **kwargs)
            results.append(result)
            return result

        monkeypatch.setattr(ml, "minimize", record_result)
        fit.optimise_hyperparameters(
            data, nb_restart=1, random_state=42,
            initial_values=np.array([0.2, 2.0] + ([1.0, 1.0] if levels else [])),
        )

        assert len(results) == 2
        for result in results:
            assert result.success, (
                f"BFGS failed: {result.message}; objective={result.fun}; "
                f"gradient={result.jac}"
            )
            assert np.linalg.norm(result.jac, ord=np.inf) <= 1e-5
        best = min(results, key=lambda result: result.fun)
        np.testing.assert_allclose(best.fun, expected["objective"], atol=1e-7, rtol=0)

        pars = fit.model.pars
        hyperparameters = np.array(
            [pars.c1, pars.c3] + ([pars.mu, pars.theta] if levels else []),
        )
        np.testing.assert_allclose(hyperparameters, ml.softplus(best.x), atol=0, rtol=0)
        np.testing.assert_allclose(
            hyperparameters, expected["hyperparameters"], atol=1e-4, rtol=0,
        )

        # Re-evaluate the committed fitted parameters, not the starting vector.
        array = data.to_numpy()
        Y, Z = construct_Y_Z(array, 2, fit.covid_indices)
        args = (
            np.log(np.expm1(hyperparameters)), array, 2, fit.covid_indices,
            fit.vars_in_levels, 0, fit.model, Y, Z,
        )
        objective = ml.objective_function(*args, True, fit.soc_, fit.sur_)
        negative_logml = ml.objective_function(*args, False, fit.soc_, fit.sur_)
        np.testing.assert_allclose(objective, expected["objective"], atol=1e-7, rtol=0)
        np.testing.assert_allclose(
            negative_logml, expected["negative_logml"], atol=1e-4, rtol=0,
        )

        fit.sample(data, N_draws=1, point_only=True, progressbar=False)
        fit.forecast(H=3, point_only=True)
        np.testing.assert_allclose(fit.beta_point, expected["beta"], atol=1e-5, rtol=0)
        np.testing.assert_allclose(fit.sigma_point, expected["sigma"], atol=1e-6, rtol=0)
        np.testing.assert_allclose(
            fit.forecast_unconditional[0, -3:], expected["forecast"], atol=1e-5, rtol=0,
        )