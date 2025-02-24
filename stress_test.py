# stress_test.py

import os
import psutil
import time
import pandas as pd
import numpy as np
from anubis import Anubis

def memory_in_mb():
    process = psutil.Process(os.getpid())
    return process.memory_info().rss / 1024**2

if __name__ == "__main__":
    # Example: load your macro data
    data_file = "/Users/alessandromassaad/RegimeDetection/warehouse/macro_reduced.parquet"
    macro_df = pd.read_parquet(data_file)
    macro_df.index = pd.to_datetime(macro_df.index)

    # Possibly reduce the dataset for speed
    macro_data = macro_df[['Macro_Signal_1', 'Macro_Signal_2']].head(500).copy()

    # Build some model combos for concurrency
    model_list = []
    for n in [2,3,4]:
        model_list.append({
            "model_type": "gaussian",
            "n": n,
            "covariance_type": "full",
            "n_iter": 10,
            "tol": 1e-3
        })
        model_list.append({
            "model_type": "gmm",
            "n": n,
            "covariance_type": "full",
            "n_mix": 2,
            "n_iter": 10,
            "tol": 1e-3
        })
        model_list.append({
            "model_type": "arhmm",
            "n": n,
            "covariance_type": "full",
            "n_iter": 10,
            "tol": 1e-3,
            "alpha": 1.0,
            "alpha_is_free": True,
            "p": 1
        })

    for test_n_jobs in [1, 2, 4]:
        print(f"\n=== STRESS TEST with n_jobs={test_n_jobs} ===")
        mem_before = memory_in_mb()
        print(f"Memory before = {mem_before:.2f} MB")

        anubis = Anubis(
            data=macro_data,
            model_list=model_list,
            out_dir="stress_test_outputs",
            n_jobs=test_n_jobs
        )

        start_time = time.time()
        anubis.run()
        end_time = time.time()

        mem_after = memory_in_mb()
        duration = end_time - start_time
        print(f"Finished in {duration:.2f}s. Memory after = {mem_after:.2f} MB")
        print(f"Memory difference = {mem_after - mem_before:.2f} MB\n")
