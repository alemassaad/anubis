# utils.py

import numpy as np
import pandas as pd
from hmm_models import GaussianHMMWrapper, GMMHMMWrapper, GaussianARHMM
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score
import math # for math.log


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
    now optionally warm-starting from a previous best model and using consistent regime labeling.
    
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
            "n": 4, # Number of regimes
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
    # Helper functions for safe metric calculation
    def safe_silhouette_score(X, labels):
        try:
            if len(np.unique(labels)) < 2 or len(X) < 2:
                return np.nan
            return silhouette_score(X, labels)
        except ValueError:
            return np.nan

    def safe_calinski_harabasz_score(X, labels):
        try:
            if len(np.unique(labels)) < 2 or len(X) < 2:
                return np.nan
            return calinski_harabasz_score(X, labels)
        except ValueError:
            return np.nan

    def safe_davies_bouldin_score(X, labels):
        try:
            if len(np.unique(labels)) < 2 or len(X) < 2:
                return np.nan
            return davies_bouldin_score(X, labels)
        except ValueError:
            return np.nan

    model_type = model_spec["model_type"]
    n_regimes_spec = model_spec["n"] # n_regimes specified for this run
    covariance_type = model_spec.get("covariance_type", "full")
    n_mix = model_spec.get("n_mix", 2)
    n_iter = model_spec.get("n_iter", 100)
    tol = model_spec.get("tol", 1e-4)
    arhmm_initial_alpha = model_spec.get("alpha", 1.0)
    alpha_is_free = model_spec.get("alpha_is_free", False)
    p = model_spec.get("p", 1)

    best_model = None
    best_logL = -np.inf

    for i in range(n_init):
        current_n_regimes_for_init = n_regimes_spec
        
        # For warm-starting parameter dimensions, n_regimes should match prev_model if possible
        # The regime assignment in .fit() will handle differences in n_regimes later.
        # Here, we ensure the GaussianARHMM is *initialized* with a number of regimes
        # that allows parameter copying if prev_model exists.
        # The n_regimes_spec is the target for the *new* model.
        # The HMMlearn wrappers (GaussianHMM, GMMHMM) have their n_components set and
        # the init_params='st' means if means_/covars_ are set (copied from prev_model),
        # they will be used. Their internal n_components is fixed at init.

        # For ARHMM, if warm-starting, the self.n_regimes of temp_model will be set by its constructor.
        # If prev_model exists, we copy params. The EM will run on temp_model.n_regimes.
        # The final n_regimes can change after assignment in ARHMM.fit().


        if model_type.lower() == 'gaussian':
            temp_model = GaussianHMMWrapper(
                n_regimes=n_regimes_spec, # Target n_regimes
                covariance_type=covariance_type,
                n_iter=n_iter,
                tol=tol
            )
            if prev_model is not None and isinstance(prev_model, GaussianHMMWrapper):
                # Ensure dimensions match for copying means/covars.
                # If n_regimes_spec is different, only copy if prev_model.n_regimes == n_regimes_spec.
                # This is a simplification; more robust would be to copy a subset/superset.
                # For now, hmmlearn models take n_components at init.
                if prev_model.model.n_components == n_regimes_spec:
                    if hasattr(prev_model.model, 'means_') and prev_model.model.means_ is not None:
                        temp_model.model.means_ = prev_model.model.means_.copy()
                        temp_model.model.means_ += 0.01 * np.random.randn(*temp_model.model.means_.shape)
                    if hasattr(prev_model.model, 'covars_') and prev_model.model.covars_ is not None:
                        temp_model.model.covars_ = prev_model.model.covars_.copy()

        elif model_type.lower() == 'gmm':
            temp_model = GMMHMMWrapper(
                n_regimes=n_regimes_spec, # Target n_regimes
                n_mix=n_mix,
                covariance_type=covariance_type,
                n_iter=n_iter,
                tol=tol
            )
            if prev_model is not None and isinstance(prev_model, GMMHMMWrapper):
                 if prev_model.model.n_components == n_regimes_spec: # Check n_regimes match
                    if hasattr(prev_model.model, 'means_') and prev_model.model.means_ is not None:
                        temp_model.model.means_ = prev_model.model.means_.copy()
                        temp_model.model.means_ += 0.01 * np.random.randn(*temp_model.model.means_.shape)
                    if hasattr(prev_model.model, 'covars_') and prev_model.model.covars_ is not None:
                        temp_model.model.covars_ = prev_model.model.covars_.copy()
                    if hasattr(prev_model.model, 'weights_') and prev_model.model.weights_ is not None:
                        temp_model.model.weights_ = prev_model.model.weights_.copy()

        elif model_type.lower() == 'arhmm':
            # ARHMM is initialized with n_regimes_spec.
            # If prev_model exists, its parameters (means_, covars_, etc.) will be copied.
            # The shapes of these copied parameters are determined by prev_model.n_regimes.
            # _init_params in ARHMM will use these if shapes are compatible or re-initialize.
            # This needs careful handling if n_regimes_spec differs from prev_model.n_regimes.
            # For simplicity now, let's assume if prev_model is used for warm-start,
            # n_regimes_spec should ideally match prev_model.n_regimes for direct param copy.
            # The regime assignment in ARHMM.fit will then handle final effective n_regimes.

            n_for_arhmm_init = n_regimes_spec
            if prev_model is not None and isinstance(prev_model, GaussianARHMM):
                 # If warm-starting, initialize with prev_model's n_regimes to match param dimensions
                 # The assignment can then decide the final n_regimes.
                 # This is a bit different from HMMlearn wrappers.
                 # Let's stick to n_regimes_spec for initialization for now, and _init_params
                 # will decide if it can use copied params.
                 pass


            temp_model = GaussianARHMM(
                n_regimes=n_for_arhmm_init, # Initialize with target number of regimes
                covariance_type=covariance_type,
                n_iter=n_iter,
                tol=tol,
                alpha=arhmm_initial_alpha,
                alpha_is_free=alpha_is_free,
                p=p
            )
            if prev_model is not None and isinstance(prev_model, GaussianARHMM):
                # Check if n_features match before copying
                n_features_train = train_data.shape[1]
                if hasattr(prev_model, 'means_') and prev_model.means_ is not None and \
                   prev_model.means_.shape[1] == n_features_train:

                    # Only copy if number of regimes for initialization matches previous model's actual regimes
                    # This ensures copied parameters have the correct dimensions for the temp_model
                    if temp_model.n_regimes == prev_model.n_regimes:
                        print(f"[fit_and_score_window] ARHMM Warm start: Copying params from prev_model with {prev_model.n_regimes} regimes.")
                        temp_model.means_ = prev_model.means_.copy()
                        temp_model.means_ += 0.01 * np.random.randn(*temp_model.means_.shape)
                        if hasattr(prev_model, 'covars_') and prev_model.covars_ is not None:
                            temp_model.covars_ = prev_model.covars_.copy()
                        if hasattr(prev_model, 'ar_coeffs_') and prev_model.ar_coeffs_ is not None:
                            temp_model.ar_coeffs_ = prev_model.ar_coeffs_.copy()
                        if hasattr(prev_model, 'transmat_') and prev_model.transmat_ is not None:
                            temp_model.transmat_ = prev_model.transmat_.copy()
                        if hasattr(prev_model, 'startprob_') and prev_model.startprob_ is not None:
                            temp_model.startprob_ = prev_model.startprob_.copy()
                        if hasattr(prev_model, 'alpha'):
                            temp_model.alpha = prev_model.alpha
                    else:
                        print(f"[fit_and_score_window] ARHMM Warm start: n_regimes mismatch. Target {temp_model.n_regimes}, Prev {prev_model.n_regimes}. No param copy.")
                else:
                    if not (hasattr(prev_model, 'means_') and prev_model.means_ is not None):
                         print(f"[fit_and_score_window] ARHMM Warm start: prev_model has no means. No param copy.")
                    elif prev_model.means_.shape[1] != n_features_train:
                         print(f"[fit_and_score_window] ARHMM Warm start: n_features mismatch. Train data {n_features_train}, Prev_model {prev_model.means_.shape[1]}. No param copy.")


        else:
            raise ValueError(f"Unknown model_type '{model_type}'")

        # Fit on the train_data
        current_prev_model_for_fit = None
        if model_type.lower() == 'gaussian':
            current_prev_model_for_fit = prev_model if isinstance(prev_model, GaussianHMMWrapper) else None
            temp_model.fit(train_data, previous_model=current_prev_model_for_fit)
        elif model_type.lower() == 'gmm':
            current_prev_model_for_fit = prev_model if isinstance(prev_model, GMMHMMWrapper) else None
            temp_model.fit(train_data, previous_model=current_prev_model_for_fit)
        elif model_type.lower() == 'arhmm':
            current_prev_model_for_fit = prev_model if isinstance(prev_model, GaussianARHMM) else None
            # Pass the previous ARHMM model to its fit method
            temp_model.fit(train_data, previous_model=current_prev_model_for_fit)
        
        try:
            ll = temp_model.score(train_data)
            if not np.isnan(ll) and ll > best_logL:
                best_logL = ll
                best_model = temp_model
        except Exception as e:
            print(f"Scoring failed for one init of {model_type}, N={n_regimes_spec}, iter {i}: {e}")
            pass

    if best_model is None:
        print(f"All {n_init} initializations failed to produce a valid model for {model_type}, N={n_regimes_spec}.")
        nan_results = {
            "ModelType": model_type, "N": n_regimes_spec,
            "IS_AIC": np.nan, "IS_BIC": np.nan, "IS_Silhouette": np.nan,
            "IS_Calinski-Harabasz": np.nan, "IS_Davies-Bouldin": np.nan,
            "IS_Perplexity": np.nan, "IS_EI": np.nan,
            "OOS_LogL": np.nan, "OOS_Silhouette": np.nan,
            "OOS_Calinski-Harabasz": np.nan, "OOS_Davies-Bouldin": np.nan,
            "OOS_Perplexity": np.nan, "OOS_EI": np.nan,
            "BestInitTrainLogL": -np.inf, "ModelObject": None
        }
        return nan_results

    # Compute final in-sample metrics using the best model
    # The number of parameters 'k' and n_regimes 'N' for reporting should reflect the state of best_model AFTER fitting and regime assignment
    
    final_n_regimes_in_model = best_model.n_regimes # This should be the updated n_regimes after assignment
    
    is_logL = best_model.score(train_data)
    n_samples_train = len(train_data)
    k = best_model.num_params() # num_params should use the final_n_regimes_in_model

    is_aic = 2 * k - 2 * is_logL if not np.isnan(is_logL) and k is not None else np.nan
    is_bic = k * math.log(n_samples_train) - 2 * is_logL if n_samples_train > 0 and not np.isnan(is_logL) and k is not None else np.nan
    
    pred_series = pd.Series(dtype=str) # Initialize to avoid UnboundLocalError
    try:
        pred_series = best_model.transform(train_data)
        pred_int_df = pred_series.str.extract(r'(\d+)').astype(int)
        pred_int = pred_int_df.squeeze(axis=0) if not pred_int_df.empty else pd.Series(dtype=int)
        if isinstance(pred_int, pd.DataFrame) and len(pred_int.columns) == 1:
            pred_int = pred_int.iloc[:,0]
        pred_int = pred_int - 1

        X_train = train_data.values
        if len(np.unique(pred_int)) > 1 and len(X_train) > 1 and not pred_int.empty:
            is_sil = safe_silhouette_score(X_train, pred_int.values)
            is_ch = safe_calinski_harabasz_score(X_train, pred_int.values)
            is_db = safe_davies_bouldin_score(X_train, pred_int.values)
        else:
            is_sil, is_ch, is_db = np.nan, np.nan, np.nan
    except RuntimeError as e: # Catch if transform is called on an unfit/empty model
        print(f"Error during in-sample transform for {model_type}, N={final_n_regimes_in_model}: {e}")
        is_sil, is_ch, is_db = np.nan, np.nan, np.nan


    is_perplexity = np.exp(-is_logL / n_samples_train) if n_samples_train > 0 and not np.isnan(is_logL) else np.nan

    def transform_low(x, gamma, c):
        if np.isnan(x): return np.nan
        return 1 / (1 + np.exp(-gamma * (x - c)))

    def transform_high(x, gamma, c):
        if np.isnan(x): return np.nan
        return 1 - 1 / (1 + np.exp(-gamma * (x - c)))

    gamma_AIC, c_AIC = 0.00002, 230000; gamma_BIC, c_BIC = 0.00002, 230000
    gamma_DB, c_DB = 20, 0.15; gamma_perp, c_perp = 0.01, 300
    gamma_sil, c_sil = 20, 0.15; gamma_CH, c_CH = 0.001, 5000

    trans_is_AIC = transform_low(is_aic, gamma_AIC, c_AIC)
    trans_is_BIC = transform_low(is_bic, gamma_BIC, c_BIC)
    trans_is_DB = transform_low(is_db, gamma_DB, c_DB)
    trans_is_perp = transform_low(is_perplexity, gamma_perp, c_perp)
    trans_is_sil = transform_high(is_sil, gamma_sil, c_sil)
    trans_is_CH = transform_high(is_ch, gamma_CH, c_CH)

    is_metrics_arr = np.array([trans_is_AIC, trans_is_BIC, trans_is_sil, trans_is_CH, trans_is_DB, trans_is_perp], dtype=np.float64)
    valid_mask = ~np.isnan(is_metrics_arr)
    is_EI = np.nan
    if valid_mask.sum() > 0:
        weights = np.ones(is_metrics_arr.shape[0])
        is_EI = np.sqrt(np.nansum(weights[valid_mask] * is_metrics_arr[valid_mask] ** 2) / np.nansum(weights[valid_mask]))

    # Out-of-sample metrics
    oos_logL, oos_sil, oos_ch, oos_db, oos_perp, oos_EI = [np.nan] * 6
    if len(oos_data) > 0:
        try:
            oos_ll_score = best_model.score(oos_data)
            if not np.isnan(oos_ll_score):
                oos_logL = oos_ll_score
                X_oos = oos_data.values
                pred_oos_series = best_model.transform(oos_data)
                pred_oos_int_df = pred_oos_series.str.extract(r'(\d+)').astype(int)
                pred_oos_int = pred_oos_int_df.squeeze(axis=0) if not pred_oos_int_df.empty else pd.Series(dtype=int)
                if isinstance(pred_oos_int, pd.DataFrame) and len(pred_oos_int.columns) == 1:
                     pred_oos_int = pred_oos_int.iloc[:,0]
                pred_oos_int = pred_oos_int - 1

                if len(np.unique(pred_oos_int)) > 1 and len(X_oos) > 1 and not pred_oos_int.empty:
                    oos_sil = safe_silhouette_score(X_oos, pred_oos_int.values)
                    oos_ch = safe_calinski_harabasz_score(X_oos, pred_oos_int.values)
                    oos_db = safe_davies_bouldin_score(X_oos, pred_oos_int.values)
                
                oos_perp = np.exp(-oos_logL / len(oos_data)) if len(oos_data) > 0 and not np.isnan(oos_logL) else np.nan

                trans_oos_DB = transform_low(oos_db, gamma_DB, c_DB)
                trans_oos_perp = transform_low(oos_perp, gamma_perp, c_perp)
                trans_oos_sil = transform_high(oos_sil, gamma_sil, c_sil)
                trans_oos_CH = transform_high(oos_ch, gamma_CH, c_CH)

                oos_metrics_arr = np.array([trans_oos_sil, trans_oos_CH, trans_oos_DB, trans_oos_perp], dtype=np.float64)
                valid_mask_oos = ~np.isnan(oos_metrics_arr)
                if valid_mask_oos.sum() > 0:
                    weights_oos = np.ones(oos_metrics_arr.shape[0])
                    oos_EI = np.sqrt(np.nansum(weights_oos[valid_mask_oos] * oos_metrics_arr[valid_mask_oos] ** 2) / np.nansum(weights_oos[valid_mask_oos]))
        except Exception as e:
            print(f"OOS scoring/transform failed for {model_type}, N={final_n_regimes_in_model}: {e}")

    return {
        "ModelType": model_type, "N": final_n_regimes_in_model, # Report the final number of regimes
        "IS_AIC": is_aic, "IS_BIC": is_bic, "IS_Silhouette": is_sil,
        "IS_Calinski-Harabasz": is_ch, "IS_Davies-Bouldin": is_db,
        "IS_Perplexity": is_perplexity, "IS_EI": is_EI,
        "OOS_LogL": oos_logL, "OOS_Silhouette": oos_sil,
        "OOS_Calinski-Harabasz": oos_ch, "OOS_Davies-Bouldin": oos_db,
        "OOS_Perplexity": oos_perp, "OOS_EI": oos_EI,
        "BestInitTrainLogL": best_logL, "ModelObject": best_model
    }