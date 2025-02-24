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
    Manages parallel fitting of different HMMs by taking a list of model specifications.
    Each specification is a dict that includes:
      - "model_type" (e.g. "gaussian", "gmm", "arhmm")
      - "n": number of hidden states
      - model-specific hyperparameters
    """

    def __init__(
        self,
        data: pd.DataFrame,
        model_list,
        out_dir='outputs',
        n_jobs=4
    ):
        """
        data : pd.DataFrame
            Observed data
        model_list : list of dict
            Each dict has 'model_type', 'n', and optional parameters
        out_dir : str
            Directory for results
        n_jobs : int
            Parallel processes
        """
        self.data = data
        self.model_list = model_list
        self.out_dir = out_dir
        self.n_jobs = n_jobs

        self.models_dir = os.path.join(self.out_dir, 'models')
        os.makedirs(self.models_dir, exist_ok=True)

        self.results_ = []

    def run(self):
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
                    'BIC': result['BIC']
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
        safe_name = display_name.replace(' ', '_')
        return safe_name + ".pkl"

    def save_results_csv(self):
        if not self.results_:
            print("No results to save.")
            return
        df = pd.DataFrame(self.results_)
        out_file = os.path.join(self.out_dir, "model_eval_results.csv")
        df.to_csv(out_file, index=False)
        print(f"[{self._timestamp()}] Results written to {out_file}")

    @staticmethod
    def _timestamp():
        return datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    @staticmethod
    def _estimate_time_remaining(elapsed, done, total):
        if done == 0:
            return "Unknown"
        avg_time = elapsed / done
        left = total - done
        est = left * avg_time
        return f"{est/60:.1f}m" if est > 60 else f"{est:.1f}s"
