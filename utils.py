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
    Original single-pass model fit for an entire dataset (no expansions).
    This function remains as-is for backward compatibility.
    
    Parameters
    ----------
    model_type : str
    n : int
    data : pd.DataFrame
    covariance_type : str
    n_mix : int
    n_iter : int
    tol : float
    alpha : float
    alpha_is_free : bool
    p : int
    
    Returns
    -------
    dict
        Single-run results with AIC, BIC, cluster metrics, etc.
    """
    if model_type.lower() == 'gaussian':
        model = GaussianHMMWrapper(n_regimes=n, covariance_type=covariance_type,
                                   n_iter=n_iter, tol=tol)
    elif model_type.lower() == 'gmm':
        model = GMMHMMWrapper(n_regimes=n, n_mix=n_mix, covariance_type=covariance_type,
                              n_iter=n_iter, tol=tol)
    elif model_type.lower() == 'arhmm':
        model = GaussianARHMM(n_regimes=n, covariance_type=covariance_type,
                              n_iter=n_iter, tol=tol, alpha=alpha,
                              alpha_is_free=alpha_is_free, p=p)
    else:
        raise ValueError(f"Unknown model_type '{model_type}'")

    model.fit(data)
    logL = model.score(data)
    n_samples = data.shape[0]
    k = model.num_params()
    aic = 2 * k - 2 * logL
    bic = k * np.log(n_samples) - 2 * logL

    pred_series = model.transform(data)
    pred_int = pred_series.str.extract(r'(\d+)').astype(int).squeeze() - 1
    X = data.values
    sil_score_ = silhouette_score(X, pred_int) if len(np.unique(pred_int)) > 1 else np.nan
    ch_score_ = calinski_harabasz_score(X, pred_int) if len(np.unique(pred_int)) > 1 else np.nan
    db_score_ = davies_bouldin_score(X, pred_int) if len(np.unique(pred_int)) > 1 else np.nan

    perplexity = np.exp(-logL / n_samples)

    # transformations for EI remain the same
    def transform_low(x, gamma, c):
        return 1 / (1 + np.exp(-gamma * (x - c)))

    def transform_high(x, gamma, c):
        return 1 - 1 / (1 + np.exp(-gamma * (x - c)))

    gamma_AIC, c_AIC = 0.00002, 230000
    gamma_BIC, c_BIC = 0.00002, 230000
    gamma_DB, c_DB = 20, 0.15
    gamma_perp, c_perp = 0.01, 300
    gamma_sil, c_sil = 20, 0.15
    gamma_CH, c_CH = 0.001, 5000

    trans_AIC = transform_low(aic, gamma_AIC, c_AIC)
    trans_BIC = transform_low(bic, gamma_BIC, c_BIC)
    trans_DB = transform_low(db_score_, gamma_DB, c_DB)
    trans_perp = transform_low(perplexity, gamma_perp, c_perp)
    trans_sil = transform_high(sil_score_, gamma_sil, c_sil)
    trans_CH = transform_high(ch_score_, gamma_CH, c_CH)

    weights = np.ones(6)
    transformed_metrics = np.array([trans_AIC, trans_BIC, trans_sil, trans_CH, trans_DB, trans_perp])
    EI = np.sqrt(np.sum(weights * transformed_metrics**2) / np.sum(weights))

    return {
        'ModelType': model_type,
        'N': n,
        'AIC': aic,
        'BIC': bic,
        'Silhouette': sil_score_,
        'Calinski-Harabasz': ch_score_,
        'Davies-Bouldin': db_score_,
        'Perplexity': perplexity,
        'EI': EI,
        'ModelObject': model
    }

def fit_and_score_window(
    train_data: pd.DataFrame,
    oos_data: pd.DataFrame,
    n_init: int,
    model_spec: dict,
    prev_model=None
) -> dict:
    """
    Fit a model with multiple random initializations on train_data, then compute IS & OOS metrics,
    now optionally warm-starting from a previous best model.
    
    Catches ValueError for all clustering metrics (Silhouette, Calinski-Harabasz, Davies-Bouldin)
    and sets them to NaN if scikit-learn throws an error due to insufficient samples or labels.
    
    Parameters
    ----------
    train_data : pd.DataFrame
        In-sample (training) data slice
    oos_data : pd.DataFrame
        Out-of-sample data slice (can be empty if training is 100%)
    n_init : int
        Number of random initializations to attempt
    model_spec : dict
        Defines the model, e.g. {
            "model_type": "gaussian"|"gmm"|"arhmm",
            "n": 4,
            "covariance_type": "full",
            "n_iter": 50,
            "tol": 1e-3,
            ...
        }
    prev_model : object, optional
        A previously fitted model (e.g., from the last iteration) to use for warm-starting.
        If None, the code does plain random initialization.
    
    Returns
    -------
    dict
        Contains:
        - "ModelType", "N"
        - "IS_*": AIC, BIC, Silhouette, CH, DB, Perplexity, EI
        - "OOS_*": LogL, Silhouette, CH, DB, Perplexity, EI
        - "BestInitTrainLogL": best train log-likelihood among the random inits
        - "ModelObject": the best-fitted model
    """
    import math
    def safe_silhouette_score(X, labels):
        try:
            return silhouette_score(X, labels)
        except ValueError:
            return np.nan

    def safe_calinski_harabasz_score(X, labels):
        try:
            return calinski_harabasz_score(X, labels)
        except ValueError:
            return np.nan

    def safe_davies_bouldin_score(X, labels):
        try:
            return davies_bouldin_score(X, labels)
        except ValueError:
            return np.nan

    model_type = model_spec["model_type"]
    n = model_spec["n"]
    covariance_type = model_spec.get("covariance_type", "full")
    n_mix = model_spec.get("n_mix", 2)
    n_iter = model_spec.get("n_iter", 100)
    tol = model_spec.get("tol", 1e-4)
    alpha = model_spec.get("alpha", 1.0)
    alpha_is_free = model_spec.get("alpha_is_free", False)
    p = model_spec.get("p", 1)

    best_model = None
    best_logL = -np.inf

    # Multiple inits: each time, see if prev_model is available to warm-start
    for _ in range(n_init):
        if model_type.lower() == 'gaussian':
            temp_model = GaussianHMMWrapper(
                n_regimes=n,
                covariance_type=covariance_type,
                n_iter=n_iter,
                tol=tol
            )
            # Warm start if we have a prev_model
            if prev_model is not None and isinstance(prev_model, GaussianHMMWrapper):
                if hasattr(prev_model.model, 'means_'):
                    temp_model.model.means_ = prev_model.model.means_.copy()
                if hasattr(prev_model.model, 'covars_'):
                    temp_model.model.covars_ = prev_model.model.covars_.copy()
                # optional small noise
                temp_model.model.means_ += 0.01 * np.random.randn(*temp_model.model.means_.shape)

        elif model_type.lower() == 'gmm':
            temp_model = GMMHMMWrapper(
                n_regimes=n,
                n_mix=n_mix,
                covariance_type=covariance_type,
                n_iter=n_iter,
                tol=tol
            )
            if prev_model is not None and isinstance(prev_model, GMMHMMWrapper):
                if hasattr(prev_model.model, 'means_'):
                    temp_model.model.means_ = prev_model.model.means_.copy()
                if hasattr(prev_model.model, 'covars_'):
                    temp_model.model.covars_ = prev_model.model.covars_.copy()
                if hasattr(prev_model.model, 'weights_'):
                    temp_model.model.weights_ = prev_model.model.weights_.copy()
                temp_model.model.means_ += 0.01 * np.random.randn(*temp_model.model.means_.shape)

        elif model_type.lower() == 'arhmm':
            temp_model = GaussianARHMM(
                n_regimes=n,
                covariance_type=covariance_type,
                n_iter=n_iter,
                tol=tol,
                alpha=alpha,
                alpha_is_free=alpha_is_free,
                p=p
            )
            # ARHMM warm start if desired:
            if prev_model is not None and isinstance(prev_model, GaussianARHMM):
                # This is more custom, e.g. copy means_, ar_coeffs_, etc.
                if hasattr(prev_model, 'means_') and prev_model.means_ is not None:
                    temp_model.means_ = prev_model.means_.copy()
                if hasattr(prev_model, 'covars_') and prev_model.covars_ is not None:
                    temp_model.covars_ = prev_model.covars_.copy()
                # optional small noise on means
                temp_model.means_ += 0.01 * np.random.randn(*temp_model.means_.shape)

        else:
            raise ValueError(f"Unknown model_type '{model_type}'")

        # Fit on the train_data
        temp_model.fit(train_data)
        ll = temp_model.score(train_data)
        if ll > best_logL:
            best_logL = ll
            best_model = temp_model

    # Compute final in-sample metrics
    is_logL = best_model.score(train_data)
    n_samples_train = len(train_data)
    k = best_model.num_params()

    is_aic = 2 * k - 2 * is_logL
    is_bic = k * math.log(n_samples_train) - 2 * is_logL if n_samples_train > 0 else np.nan

    pred_series = best_model.transform(train_data)
    pred_int = pred_series.str.extract(r'(\d+)').astype(int).squeeze() - 1
    X_train = train_data.values

    if len(np.unique(pred_int)) > 1 and len(X_train) > 1:
        is_sil = safe_silhouette_score(X_train, pred_int)
        is_ch = safe_calinski_harabasz_score(X_train, pred_int)
        is_db = safe_davies_bouldin_score(X_train, pred_int)
    else:
        is_sil = np.nan
        is_ch = np.nan
        is_db = np.nan

    is_perplexity = np.exp(-is_logL / n_samples_train) if n_samples_train > 0 else np.nan

    def transform_low(x, gamma, c):
        return 1 / (1 + np.exp(-gamma * (x - c)))

    def transform_high(x, gamma, c):
        return 1 - 1 / (1 + np.exp(-gamma * (x - c)))

    gamma_AIC, c_AIC = 0.00002, 230000
    gamma_BIC, c_BIC = 0.00002, 230000
    gamma_DB, c_DB = 20, 0.15
    gamma_perp, c_perp = 0.01, 300
    gamma_sil, c_sil = 20, 0.15
    gamma_CH, c_CH = 0.001, 5000

    trans_is_AIC = transform_low(is_aic, gamma_AIC, c_AIC)
    trans_is_BIC = transform_low(is_bic, gamma_BIC, c_BIC) if not np.isnan(is_bic) else np.nan
    trans_is_DB = transform_low(is_db, gamma_DB, c_DB) if not np.isnan(is_db) else np.nan
    trans_is_perp = transform_low(is_perplexity, gamma_perp, c_perp) if not np.isnan(is_perplexity) else np.nan
    trans_is_sil = transform_high(is_sil, gamma_sil, c_sil) if not np.isnan(is_sil) else np.nan
    trans_is_CH = transform_high(is_ch, gamma_CH, c_CH) if not np.isnan(is_ch) else np.nan

    is_metrics_arr = np.array([
        trans_is_AIC,
        trans_is_BIC,
        trans_is_sil,
        trans_is_CH,
        trans_is_DB,
        trans_is_perp
    ], dtype=np.float64)

    valid_mask = ~np.isnan(is_metrics_arr)
    if valid_mask.sum() == 0:
        is_EI = np.nan
    else:
        weights = np.ones(is_metrics_arr.shape[0])
        is_EI = np.sqrt(
            np.nansum(weights[valid_mask] * is_metrics_arr[valid_mask] ** 2)
            / np.nansum(weights[valid_mask])
        )

    # Out-of-sample metrics
    if len(oos_data) == 0:
        oos_logL = None
        oos_sil = None
        oos_ch = None
        oos_db = None
        oos_perp = None
        oos_EI = None
    else:
        oos_ll = best_model.score(oos_data)
        X_oos = oos_data.values
        pred_oos_series = best_model.transform(oos_data)
        pred_oos_int = pred_oos_series.str.extract(r'(\d+)').astype(int).squeeze() - 1

        if len(np.unique(pred_oos_int)) > 1 and len(X_oos) > 1:
            oos_sil = safe_silhouette_score(X_oos, pred_oos_int)
            oos_ch = safe_calinski_harabasz_score(X_oos, pred_oos_int)
            oos_db = safe_davies_bouldin_score(X_oos, pred_oos_int)
        else:
            oos_sil = np.nan
            oos_ch = np.nan
            oos_db = np.nan

        oos_perp = np.exp(-oos_ll / len(oos_data)) if len(oos_data) > 0 else np.nan

        trans_oos_DB = transform_low(oos_db, gamma_DB, c_DB) if not np.isnan(oos_db) else np.nan
        trans_oos_perp = transform_low(oos_perp, gamma_perp, c_perp) if not np.isnan(oos_perp) else np.nan
        trans_oos_sil = transform_high(oos_sil, gamma_sil, c_sil) if not np.isnan(oos_sil) else np.nan
        trans_oos_CH = transform_high(oos_ch, gamma_CH, c_CH) if not np.isnan(oos_ch) else np.nan

        oos_metrics_arr = np.array([
            trans_oos_sil,
            trans_oos_CH,
            trans_oos_DB,
            trans_oos_perp
        ], dtype=np.float64)

        valid_mask_oos = ~np.isnan(oos_metrics_arr)
        if valid_mask_oos.sum() == 0:
            oos_EI = np.nan
        else:
            weights_oos = np.ones(oos_metrics_arr.shape[0])
            oos_EI = np.sqrt(
                np.nansum(weights_oos[valid_mask_oos] * oos_metrics_arr[valid_mask_oos] ** 2)
                / np.nansum(weights_oos[valid_mask_oos])
            )

        oos_logL = oos_ll

    return {
        "ModelType": model_type,
        "N": n,

        "IS_AIC": is_aic,
        "IS_BIC": is_bic,
        "IS_Silhouette": is_sil,
        "IS_Calinski-Harabasz": is_ch,
        "IS_Davies-Bouldin": is_db,
        "IS_Perplexity": is_perplexity,
        "IS_EI": is_EI,

        "OOS_LogL": oos_logL,
        "OOS_Silhouette": oos_sil,
        "OOS_Calinski-Harabasz": oos_ch,
        "OOS_Davies-Bouldin": oos_db,
        "OOS_Perplexity": oos_perp,
        "OOS_EI": oos_EI,

        "BestInitTrainLogL": best_logL,
        "ModelObject": best_model
    }
