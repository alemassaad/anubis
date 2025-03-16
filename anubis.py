# anubis.py

import os
import time
import pickle
import numpy as np
import pandas as pd
from datetime import datetime
from concurrent.futures import ProcessPoolExecutor, as_completed

from utils import fit_and_score

class Anubis:
    """
    Manages parallel fitting of different Hidden Markov Models (HMMs) with various specifications.
    
    Named after the Egyptian god Anubis who weighed and judged the hearts of the deceased, 
    this class similarly weighs and evaluates different models to determine the best one. 
    
    The Anubis class handles the orchestration of fitting multiple HMM models in parallel,
    evaluating their performance, and saving results. It's designed for time series regime detection
    using various types of HMMs (Gaussian, GMM, ARHMM) with different hyperparameters.
    
    Parameters
    ----------
    data : pd.DataFrame
        Time series data to fit models to. Each row represents a time point, 
        and each column represents a different variable/feature.
    model_list : list of dict
        List of model specifications to fit. Each dict must contain:
        - "model_type" (str): Type of model ("gaussian", "gmm", "arhmm")
        - "n" (int): Number of hidden states (regimes)
        - Additional model-specific parameters like "covariance_type", "n_mix", etc.
    out_dir : str, default='outputs'
        Directory where results and fitted models will be saved
    n_jobs : int, default=4
        Number of parallel processes to use for model fitting
        
    Attributes
    ----------
    data : pd.DataFrame
        The input time series data
    model_list : list of dict
        List of model specifications
    out_dir : str
        Output directory path
    n_jobs : int
        Number of parallel processes
    models_dir : str
        Directory for saving fitted model objects
    results_ : list
        List that stores results from each model after fitting
        
    Examples
    --------
    >>> models_to_fit = [
    ...     {"model_type": "gaussian", "n": 3, "covariance_type": "full"},
    ...     {"model_type": "arhmm", "n": 2, "covariance_type": "diag", "alpha": 0.8}
    ... ]
    >>> runner = Anubis(data=my_time_series, model_list=models_to_fit, n_jobs=2)
    >>> runner.run()
    """



    def __init__(self, data: pd.DataFrame, model_list, out_dir='outputs', n_jobs=4):
        self.data = data
        self.model_list = model_list
        self.out_dir = out_dir
        self.n_jobs = n_jobs

        self.models_dir = os.path.join(self.out_dir, 'models')
        os.makedirs(self.models_dir, exist_ok=True)

        self.results_ = []




    def run(self):
        """
        Execute the model fitting process in parallel.
        
        This method:
        1. Submits each model specification as a parallel job
        2. Collects results as jobs complete
        3. Saves fitted models to disk
        4. Tracks progress and estimates completion time
        5. Saves a summary CSV of all model results
        
        The method uses ProcessPoolExecutor to run multiple model fits in parallel.
        Progress updates and errors are printed to the console.
        
        Returns
        -------
        None
            Results are stored in self.results_ and written to disk
        """

        tasks = self.model_list
        total_jobs = len(tasks)
        start_time = time.time()
        print(f"[{self._timestamp()}] Starting Anubis: {total_jobs} total fits...")

        with ProcessPoolExecutor(max_workers=self.n_jobs) as executor:
            futures = {}
            for idx, spec in enumerate(tasks, start=1):
                display_name = self._make_model_name(spec)
                print(f"[{self._timestamp()}] Submitting job {idx}/{total_jobs}: {display_name}")

                fut = executor.submit(
                    fit_and_score,
                    data=self.data,
                    **spec
                )
                futures[fut] = (idx, display_name)

            for fut in as_completed(futures):
                idx, display_name = futures[fut]
                try:
                    result = fut.result()
                except Exception as exc:
                    print(f"[{self._timestamp()}] ERROR in job {idx}: {display_name}: {str(exc)}")
                    continue

                # Store numeric results
                self.results_.append({
                    'ModelType': result['ModelType'],
                    'N': result['N'],
                    'AIC': result['AIC'],
                    'BIC': result['BIC'],
                    'Silhouette': result['Silhouette'],
                    'Calinski-Harabasz': result['Calinski-Harabasz'],
                    'Davies-Bouldin': result['Davies-Bouldin'],
                    'Perplexity': result['Perplexity'],
                    'EI': result['EI']
                })

                # Save model object
                model_filename = self._make_model_filename(result['ModelType'], result['N'], display_name)
                model_path = os.path.join(self.models_dir, model_filename)
                with open(model_path, 'wb') as f:
                    pickle.dump(result['ModelObject'], f)

                elapsed = time.time() - start_time
                done_so_far = len(self.results_)
                eta_str = self._estimate_time_remaining(elapsed, done_so_far, total_jobs)
                print(
                    f"[{self._timestamp()}] Finished job {idx}/{total_jobs}: {display_name}. "
                    f"Elapsed={elapsed:.1f}s, Remaining={eta_str}"
                )

        self.save_results_csv()
        print(f"[{self._timestamp()}] All tasks complete. Results saved to {self.out_dir}.")




    def _make_model_name(self, spec: dict) -> str:
        """
        Generate a descriptive name for a model based on its specification.
        
        Creates a human-readable string representing the model configuration,
        which is used for display purposes and as part of filenames.
        
        Parameters
        ----------
        spec : dict
            Model specification dictionary containing model_type, n, and other hyperparameters
            
        Returns
        -------
        str
            A formatted string name like "gaussian_N=3_cov=full" or 
            "arhmm_N=2_cov=diag_alpha=0.80_p=1"
            
        Notes
        -----
        Different model types have different naming formats:
        - Gaussian HMM: gaussian_N=<n>_cov=<covariance_type>
        - GMM HMM: gmm_N=<n>_cov=<covariance_type>_mix=<n_mix>
        - AR HMM: arhmm_N=<n>_cov=<covariance_type>_alpha=<alpha>_p=<p>
        """
        
        model_type = spec.get("model_type", "unknown").lower()
        n = spec.get("n", "?")
        cov = spec.get("covariance_type", "NA")

        if model_type == "gaussian":
            return f"gaussian_N={n}_cov={cov}"
        elif model_type == "gmm":
            nmix = spec.get("n_mix", "NA")
            return f"gmm_N={n}_cov={cov}_mix={nmix}"
        elif model_type == "arhmm":
            alpha = spec.get("alpha", 1.0)
            p = spec.get("p", 1)
            alpha_is_free = spec.get("alpha_is_free", False)
            name = f"arhmm_N={n}_cov={cov}_alpha={alpha:.2f}_p={p}"
            if alpha_is_free:
                name += "_afree"
            return name
        else:
            return f"{model_type}_N={n}"

    def _make_model_filename(self, model_type: str, n: int, display_name: str) -> str:
        """
        Generate a filename for saving a fitted model object.
        
        Parameters
        ----------
        model_type : str
            Type of model ("gaussian", "gmm", "arhmm")
        n : int
            Number of hidden states
        display_name : str
            Human-readable model name from _make_model_name()
            
        Returns
        -------
        str
            Filename for the model, e.g., "gaussian_N=3_cov=full.pkl"
        """

        safe_name = display_name.replace(' ', '_')
        return safe_name + ".pkl"

    def save_results_csv(self):
        """
        Save the model evaluation results to a CSV file.
        
        Writes the contents of self.results_ (list of dictionaries with model evaluation 
        metrics) to a CSV file named "model_eval_results.csv" in the output directory.
        
        The CSV includes columns for model type, number of states, and various evaluation
        metrics (AIC, BIC, Silhouette, etc.).
        
        Returns
        -------
        None
            File is written to disk at {self.out_dir}/model_eval_results.csv
        """

        if not self.results_:
            print("No results to save.")
            return
        df = pd.DataFrame(self.results_)
        out_file = os.path.join(self.out_dir, "model_eval_results.csv")
        df.to_csv(out_file, index=False)
        print(f"[{self._timestamp()}] Results written to {out_file}")

    @staticmethod
    def _timestamp():
        """
        Generate a formatted timestamp string.
        
        Used for logging and progress messages.
        
        Returns
        -------
        str
            Current timestamp in format "YYYY-MM-DD HH:MM:SS"
        """

        return datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    @staticmethod
    def _estimate_time_remaining(elapsed, done, total):
        """
        Estimate the remaining time to complete all model fits.
        
        Parameters
        ----------
        elapsed : float
            Time elapsed so far in seconds
        done : int
            Number of completed model fits
        total : int
            Total number of model fits to perform
            
        Returns
        -------
        str
            Formatted string with estimated time remaining (e.g., "5.2m" or "45.3s")
        
        Notes
        -----
        Uses simple average time per job to estimate remaining time.
        Returns time in minutes if > 60 seconds, otherwise in seconds.
        """

        if done == 0:
            return "Unknown"
        avg_time = elapsed / done
        left = total - done
        est = left * avg_time
        return f"{est/60:.1f}m" if est > 60 else f"{est:.1f}s"
