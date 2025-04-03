# anubis.py

import os
import time
import pickle
import numpy as np
import pandas as pd
from datetime import datetime
from concurrent.futures import ProcessPoolExecutor, as_completed

from utils import fit_and_score_window

class Anubis:
    """
    Manages parallel fitting of HMMs using a two-level approach:
    1) Parallelization across different model specs (gaussian, gmm, arhmm, etc.)
    2) Within each model spec, expansions proceed sequentially from
       `initial_train_frac` up to 100%, warm-starting from the last iteration.
    
    Named after the Egyptian god Anubis who weighed and judged hearts, 
    this class similarly weighs and evaluates HMM models to determine the best ones.
    
    Key Features
    ------------
    1. Two-Level Parallel:
       - Each model spec runs in its own process, so different specs do not block each other.
       - Inside each model, expansions are strictly sequential with warm starts.
    2. Expanding-window approach (per model):
       - Start with `initial_train_frac` of the data.
       - Increment by `increment_frac` each iteration, up to 100%.
       - Remainder is OOS at each step for separate OOS metrics.
    3. Multiple random inits:
       - For each window, up to `n_init` attempts. The best log-likelihood solution is retained.
    4. Warm-Starting:
       - The best solution from iteration t is used to initialize iteration t+1 within the same spec.
    5. Logging & CSV output:
       - All results are aggregated in self.results_, then written to "expanding_eval_results.csv".
       - Each fitted model (best at each iteration) is saved to ./outputs/models

    Parameters
    ----------
    data : pd.DataFrame
        Time series data, rows = time points, columns = features
    model_list : list of dict
        List of model specifications. Each dict must define:
          - "model_type": e.g., "gaussian", "gmm", or "arhmm"
          - "n": number of hidden states
          - Additional hyperparams: e.g., "covariance_type", "n_mix", "alpha", etc.
    out_dir : str, optional
        Output directory for CSV & model pickles, default="outputs"
    n_jobs : int, optional
        Number of parallel processes for model-level concurrency, default=4
    initial_train_frac : float, optional
        Initial fraction of data used in the first expansion (0..1), default=0.6
    increment_frac : float, optional
        Fraction by which the training set grows each iteration, default=0.1
    n_init : int, optional
        Number of random inits for each expansion, default=2

    Attributes
    ----------
    data : pd.DataFrame
        The entire dataset
    model_list : list of dict
        List of model specs
    out_dir : str
        Where results & pickles are saved
    n_jobs : int
        Model-level concurrency
    models_dir : str
        Directory to save pickled models
    initial_train_frac : float
        Starting fraction for expansions
    increment_frac : float
        Growth fraction each step
    n_init : int
        Random inits per step
    results_ : list
        Holds the aggregated metrics across all expansions & specs
    """

    def __init__(
        self, 
        data: pd.DataFrame, 
        model_list, 
        out_dir='outputs', 
        n_jobs=4,
        initial_train_frac=0.6,
        increment_frac=0.1,
        n_init=2
    ):
        self.data = data
        self.model_list = model_list
        self.out_dir = out_dir
        self.n_jobs = n_jobs

        self.initial_train_frac = initial_train_frac
        self.increment_frac = increment_frac
        self.n_init = n_init

        self.models_dir = os.path.join(self.out_dir, 'models')
        os.makedirs(self.models_dir, exist_ok=True)

        self.results_ = []
        # We don't strictly need last_best_models now, but let's keep it
        # for legacy usage. We'll not remove it to preserve old functionality:
        self.last_best_models = {}

    def run_single_model_expansions(self, spec: dict):
        """
        Executes a strictly sequential expanding-window loop for one model spec.
        
        Parameters
        ----------
        spec : dict
            A single model specification, e.g. {"model_type":"gaussian", "n":3, ...}
        
        Returns
        -------
        list
            A list of result rows (dicts) with in-sample and out-of-sample metrics
            for each expansion step, to be merged into self.results_ in the main run.
        
        Notes
        -----
        1. We build expansions from initial_train_frac up to 100% in increments.
        2. We warm-start from the best model of the previous iteration for each step.
        3. Each iteration calls fit_and_score_window(...) once, with multiple inits.
        4. We log each step for clarity.
        """
        model_name = self._make_model_name(spec)
        n_samples = len(self.data)

        print(f"[{self._timestamp()}] [Spec={model_name}] Starting expansions with data of size={n_samples}.")

        # Build expansions list
        expansions = []
        train_size = int(n_samples * self.initial_train_frac)
        while train_size <= n_samples:
            expansions.append(train_size)
            if train_size == n_samples:
                break
            train_size += int(n_samples * self.increment_frac)
            if train_size > n_samples:
                train_size = n_samples

        local_results = []
        local_best_model = None  # warm-start from here

        for idx, tsize in enumerate(expansions, start=1):
            train_data = self.data.iloc[:tsize]
            oos_data = self.data.iloc[tsize:]

            print(f"[{self._timestamp()}] [Spec={model_name}] Iteration {idx}/{len(expansions)}: TrainSize={tsize}. Fitting...")

            # Fit with possible warm start:
            result_dict = fit_and_score_window(
                train_data=train_data,
                oos_data=oos_data,
                n_init=self.n_init,
                model_spec=spec,
                prev_model=local_best_model  # warm-start
            )

            local_best_model = result_dict["ModelObject"]  # update best model for next iteration

            # Build a row with iteration info:
            row = {
                'Iteration': idx,
                'TrainSize': tsize,
                'TrainFraction': round(tsize / n_samples, 4),
                'ModelType': result_dict['ModelType'],
                'N': result_dict['N'],

                # In-sample metrics
                'IS_AIC': result_dict['IS_AIC'],
                'IS_BIC': result_dict['IS_BIC'],
                'IS_Silhouette': result_dict['IS_Silhouette'],
                'IS_Calinski-Harabasz': result_dict['IS_Calinski-Harabasz'],
                'IS_Davies-Bouldin': result_dict['IS_Davies-Bouldin'],
                'IS_Perplexity': result_dict['IS_Perplexity'],
                'IS_EI': result_dict['IS_EI'],

                # OOS metrics
                'OOS_LogL': result_dict['OOS_LogL'],
                'OOS_Silhouette': result_dict['OOS_Silhouette'],
                'OOS_Calinski-Harabasz': result_dict['OOS_Calinski-Harabasz'],
                'OOS_Davies-Bouldin': result_dict['OOS_Davies-Bouldin'],
                'OOS_Perplexity': result_dict['OOS_Perplexity'],
                'OOS_EI': result_dict['OOS_EI'],

                # For reference: best train LL among n_init
                'BestInitTrainLogL': result_dict['BestInitTrainLogL']
            }
            local_results.append(row)

            # Save the newly chosen best model object to disk
            model_filename = f"{model_name}_iter{idx}_tsize{tsize}.pkl"
            model_path = os.path.join(self.models_dir, model_filename)
            with open(model_path, 'wb') as f:
                pickle.dump(local_best_model, f)

            print(f"[{self._timestamp()}] [Spec={model_name}] Finished iteration {idx}/{len(expansions)}. (TrainSize={tsize})")

        print(f"[{self._timestamp()}] [Spec={model_name}] All expansions complete.")
        return local_results

    def run(self):
        """
        Top-level run: 
        1) Parallelize across model specs. Each spec is handled by run_single_model_expansions(...) 
           in its own process.
        2) Aggregates results from each spec into self.results_, then writes CSV.
        
        Returns
        -------
        None
            Results are in self.results_ and also saved to CSV.
        
        Notes
        -----
        - If you have e.g. 5 model specs, each one runs expansions sequentially
          in a separate process. So if one is slow (like ARHMM), it won't block 
          others from finishing.
        - Inside run_single_model_expansions, each iteration warm-starts from the last.
        """
        if len(self.data) == 0:
            print("[WARNING] The dataset is empty. No fitting performed.")
            return

        print(f"[{self._timestamp()}] Starting parallel runs for {len(self.model_list)} model specs.")

        # We'll parallelize one spec per process.
        from concurrent.futures import ProcessPoolExecutor, as_completed
        with ProcessPoolExecutor(max_workers=self.n_jobs) as executor:
            future_map = {}
            for spec in self.model_list:
                # Submit one job per spec
                fut = executor.submit(self.run_single_model_expansions, spec)
                future_map[fut] = spec

            # Gather results
            for fut in as_completed(future_map):
                spec = future_map[fut]
                spec_key = self._make_model_name(spec)
                try:
                    spec_results = fut.result()
                except Exception as exc:
                    print(f"[{self._timestamp()}] ERROR for spec={spec_key}: {exc}")
                    spec_results = []
                # Add to self.results_
                self.results_.extend(spec_results)

        # Now write everything to CSV
        self.save_results_csv(final_filename="expanding_eval_results.csv")
        print(f"[{self._timestamp()}] All expansions complete across all specs. Results saved to {self.out_dir}.")

    def _make_model_name(self, spec: dict) -> str:
        """
        Generate a descriptive string for a given model specification.
        
        Parameters
        ----------
        spec : dict
            Model specification containing keys like 'model_type', 'n', etc.
            
        Returns
        -------
        str
            A formatted string (e.g. "gaussian_N=3_cov=full")
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

    def save_results_csv(self, final_filename="model_eval_results.csv"):
        """
        Save the collected results to a CSV file in the output directory.
        
        Parameters
        ----------
        final_filename : str
            Name of the CSV file. Default="model_eval_results.csv"
        
        Returns
        -------
        None
            Writes self.results_ to {out_dir}/{final_filename}.
        """
        if not self.results_:
            print("No results to save.")
            return
        df = pd.DataFrame(self.results_)
        out_file = os.path.join(self.out_dir, final_filename)
        df.to_csv(out_file, index=False)
        print(f"[{self._timestamp()}] Results written to {out_file}")

    @staticmethod
    def _timestamp():
        """
        Generate a formatted timestamp string for progress messages.
        
        Returns
        -------
        str
            Timestamp in "YYYY-MM-DD HH:MM:SS" format.
        """
        from datetime import datetime
        return datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    @staticmethod
    def _estimate_time_remaining(elapsed, done, total):
        """
        Estimate time remaining using the average time per completed job.
        
        Parameters
        ----------
        elapsed : float
            Seconds elapsed so far
        done : int
            Number of completed tasks
        total : int
            Total tasks
            
        Returns
        -------
        str
            Time estimate in seconds or minutes
        """
        if done == 0:
            return "Unknown"
        avg_time = elapsed / done
        left = total - done
        est = left * avg_time
        return f"{est/60:.1f}m" if est > 60 else f"{est:.1f}s"
