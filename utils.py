# utils.py

import numpy as np
import pandas as pd
from hmm_models import GaussianHMMWrapper, GMMHMMWrapper, GaussianARHMM

def fit_and_score(
    model_type,
    n,
    data: pd.DataFrame,
    covariance_type='full',
    n_mix=2,
    n_iter=100,
    tol=1e-4,
    alpha=1.0,
    alpha_is_free=False,
    p=1
):
    """
    Fits one of the wrapper classes (GaussianHMMWrapper, GMMHMMWrapper, GaussianARHMM),
    computes log-likelihood, AIC, and BIC, and returns a dict with results.
    """
    if model_type.lower() == 'gaussian':
        model = GaussianHMMWrapper(
            n_regimes=n,
            covariance_type=covariance_type,
            n_iter=n_iter,
            tol=tol
        )
    elif model_type.lower() == 'gmm':
        model = GMMHMMWrapper(
            n_regimes=n,
            n_mix=n_mix,
            covariance_type=covariance_type,
            n_iter=n_iter,
            tol=tol
        )
    elif model_type.lower() == 'arhmm':
        model = GaussianARHMM(
            n_regimes=n,
            covariance_type=covariance_type,
            n_iter=n_iter,
            tol=tol,
            alpha=alpha,
            alpha_is_free=alpha_is_free,
            p=p
        )
    else:
        raise ValueError(f"Unknown model_type '{model_type}'")

    model.fit(data)
    logL = model.score(data)

    # Number of data points
    n_samples = data.shape[0]

    # Parameter count
    k = model.num_params()

    # AIC & BIC
    aic = 2*k - 2*logL
    bic = k*np.log(n_samples) - 2*logL

    return {
        'ModelType': model_type,
        'N': n,
        'AIC': aic,
        'BIC': bic,
        'ModelObject': model
    }
