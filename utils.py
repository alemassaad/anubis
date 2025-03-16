# utils.py

import numpy as np
import pandas as pd
from hmm_models import GaussianHMMWrapper, GMMHMMWrapper, GaussianARHMM
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score

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
    Fit a specified HMM model to data and compute various evaluation metrics.
    
    This function handles model creation, fitting, and evaluation. It's the core
    function used by the Anubis class to evaluate individual model configurations.
    
    Parameters
    ----------
    model_type : str
        Type of model to fit, one of 'gaussian', 'gmm', or 'arhmm'
    n : int
        Number of hidden states (regimes)
    data : pd.DataFrame
        Time series data to fit
    covariance_type : str, default='full'
        Type of covariance matrices, one of 'full', 'diag'
    n_mix : int, default=2
        Number of mixture components (only for 'gmm' model type)
    n_iter : int, default=100
        Maximum EM iterations
    tol : float, default=1e-4
        Convergence tolerance
    alpha : float, default=1.0
        Initial global AR coefficient (only for 'arhmm' model type)
    alpha_is_free : bool, default=False
        Whether to learn alpha (only for 'arhmm' model type)
    p : int, default=1
        AR order (only for 'arhmm' model type)
        
    Returns
    -------
    dict
        Dictionary containing:
        - 'ModelType': model type string
        - 'N': number of states
        - 'AIC': Akaike Information Criterion
        - 'BIC': Bayesian Information Criterion
        - 'Silhouette': Silhouette coefficient for clustering quality
        - 'Calinski-Harabasz': Calinski-Harabasz index
        - 'Davies-Bouldin': Davies-Bouldin index
        - 'Perplexity': Perplexity score
        - 'EI': Explainability Index - an aggregate metric
        - 'ModelObject': the fitted model object
        
    Notes
    -----
    The Explainability Index (EI) is a custom metric that combines the other metrics
    using sigmoid transformations to normalize them to a common scale.
    
    For the clustering metrics (Silhouette, Calinski-Harabasz, Davies-Bouldin),
    the function first transforms the model predictions to integer labels.
    
    Examples
    --------
    >>> result = fit_and_score('gaussian', n=3, data=my_data, covariance_type='full')
    >>> print(f"AIC: {result['AIC']}, BIC: {result['BIC']}")
    >>> best_model = result['ModelObject']
    """

    # --- Fit the HMM model based on model_type ---
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

    # Fit the model and compute basic metrics
    model.fit(data)
    logL = model.score(data)
    n_samples = data.shape[0]
    k = model.num_params()
    aic = 2 * k - 2 * logL
    bic = k * np.log(n_samples) - 2 * logL

    # Compute clustering metrics based on state predictions
    pred_series = model.transform(data)  # returns a pd.Series like "regime1", "regime2", ...
    pred_int = pred_series.str.extract(r'(\d+)').astype(int).squeeze() - 1
    X = data.values
    sil_score = silhouette_score(X, pred_int) if len(np.unique(pred_int)) > 1 else np.nan
    ch_score = calinski_harabasz_score(X, pred_int) if len(np.unique(pred_int)) > 1 else np.nan
    db_score = davies_bouldin_score(X, pred_int) if len(np.unique(pred_int)) > 1 else np.nan

    perplexity = np.exp(-logL / n_samples)

    # ------------------- Explainability Index (EI) Implementation -------------------
    # The goal is to transform each metric to a value between 0 and 1 so that lower values indicate better performance.
    # For metrics where "lower is better" (AIC, BIC, DB, Perplexity), we use a sigmoid transformation (F_low):
    #   F_low(x) = 1/(1 + exp(-gamma*(x - c)))
    # For metrics where "higher is better" (Silhouette, Calinski-Harabasz), we want lower transformed values when x is high.
    # We achieve this by inverting the sigmoid:
    #   F_high(x) = 1 - 1/(1 + exp(-gamma*(x - c)))
    # The parameters gamma and c are set as defaults based on our expected ranges.
    
    # Define transformation functions:
    def transform_low(x, gamma, c):
        """Transform a metric where lower is better."""
        return 1 / (1 + np.exp(-gamma * (x - c)))
    
    def transform_high(x, gamma, c):
        """Transform a metric where higher is better (inverted so that lower is better)."""
        return 1 - 1 / (1 + np.exp(-gamma * (x - c)))
    
    # Set default tuning parameters (these may be tuned further):
    # For AIC (typical range ~218k to 240k, ideal lower ~230k)
    gamma_AIC, c_AIC = 0.00002, 230000
    # For BIC (similar range)
    gamma_BIC, c_BIC = 0.00002, 230000
    # For Davies-Bouldin (typical range ~0.04 to 0.22, ideal lower ~0.15)
    gamma_DB, c_DB = 20, 0.15
    # For Perplexity (typical range ~190 to 490, ideal lower ~300)
    gamma_perp, c_perp = 0.01, 300
    # For Silhouette (range ~0.04 to 0.22, ideal higher; so use transform_high with ideal ~0.15)
    gamma_sil, c_sil = 20, 0.15
    # For Calinski-Harabasz (range ~2000 to ~7200, ideal higher; choose ideal ~5000)
    gamma_CH, c_CH = 0.001, 5000
    
    # Apply the transformations:
    trans_AIC = transform_low(aic, gamma_AIC, c_AIC)
    trans_BIC = transform_low(bic, gamma_BIC, c_BIC)
    trans_DB = transform_low(db_score, gamma_DB, c_DB)
    trans_perp = transform_low(perplexity, gamma_perp, c_perp)
    trans_sil = transform_high(sil_score, gamma_sil, c_sil)
    trans_CH = transform_high(ch_score, gamma_CH, c_CH)
    
    # Use equal weights for all six metrics:
    weights = np.ones(6)
    transformed_metrics = np.array([trans_AIC, trans_BIC, trans_sil, trans_CH, trans_DB, trans_perp])
    
    # Compute EI as the weighted Euclidean norm of the transformed metrics:
    EI = np.sqrt(np.sum(weights * transformed_metrics**2) / np.sum(weights))



    return {
        'ModelType': model_type,
        'N': n,
        'AIC': aic,
        'BIC': bic,
        'Silhouette': sil_score,  
        'Calinski-Harabasz': ch_score,
        'Davies-Bouldin': db_score,
        'Perplexity': perplexity,
        'EI': EI,  # Newly added Explainability Index
        'ModelObject': model
    }
