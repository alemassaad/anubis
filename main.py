# main.py

import pandas as pd
from anubis import Anubis

if __name__ == "__main__":
    data_file = "/Users/alessandromassaad/RegimeDetection/warehouse/macro_reduced.parquet"
    macro_df = pd.read_parquet(data_file)
    macro_df.index = pd.to_datetime(macro_df.index)

    macro_data = macro_df[['Macro_Signal_1', 'Macro_Signal_2']].copy()

    n = 4
    N = 8
    
    model_list = (
        [
            {
                "model_type": "gaussian",
                "n": k,
                "covariance_type": "full",
                "n_iter": 30,
                "tol": 1e-4
            }
            for k in range(n, N+1)
        ]
        + [
            {
                "model_type": "gmm",
                "n": k,
                "covariance_type": "full",
                "n_mix": 2,
                "n_iter": 30,
                "tol": 1e-4
            }
            for k in range(n, N+1)
        ]
        + [
            {
                "model_type": "arhmm",
                "n": k,
                "covariance_type": "full",
                "n_iter": 20,
                "tol": 1e-3,
                "alpha": 1.0,
                "alpha_is_free": False, 
                "p": 1
            }
            for k in range(n, N+1)
        ]
    )

    # Create an Anubis instance with only data, model_list, etc.
    anubis = Anubis(
        data=macro_data,
        model_list=model_list,
        out_dir='outputs',
        n_jobs=4
    )

    anubis.run()
    # Results: 'outputs/model_eval_results.csv' plus pickled models in 'outputs/models/'
