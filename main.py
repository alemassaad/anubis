# main.py

import pandas as pd
from anubis import Anubis
from clean_outputs import clean_outputs 

if __name__ == "__main__":
    # First, clean outputs directory with confirmation
    clean_outputs()
    
    data_file = "/Users/alessandromassaad/Desktop/ASK2Ai/RegimeDetection/warehouse/macro_reduced.parquet"
    macro_df = pd.read_parquet(data_file)
    macro_df.index = pd.to_datetime(macro_df.index)

    macro_data = macro_df[['Macro_Signal_1', 'Macro_Signal_2']].copy()





    n = 2
    N = 6 # (included)
    
    
    
    # # Gaussian HMM ONLY
    # model_list = [
    #                 {"model_type": "gaussian", "n": k, "covariance_type": "full", "n_iter": 5, "tol": 1e-2}
    #                 for k in range(n, N+1)
    #                 ]




    # # GAUSSIAN MIXTURE HMM ONLY
    # model_list = [
    #                 {"model_type": "gmm", "n": k, "covariance_type": "full", "n_iter": 5, "tol": 1e-2, "n_mix": 3}
    #                 for k in range(n, N+1)
    #                 ]
    
    
    
    
    
    
    
    # GaussianARHMM ONLY
    model_list = [
                    {
                        "model_type": "arhmm", 
                        "n": k, 
                        "covariance_type": "full", 
                        "n_iter": 5, 
                        "tol": 1e-2
                    }
                for k in range(n, N+1)
                ]
    






    # Here we set up our expanding-window parameters:
    initial_train_frac = 0.5
    increment_frac = 0.1
    n_init = 12  # multiple random starts

    # Create Anubis instance for the expanding-window approach:
    anubis = Anubis(
        data=macro_data,
        model_list=model_list,
        out_dir='outputs',
        n_jobs=4,
        initial_train_frac=initial_train_frac,
        increment_frac=increment_frac,
        n_init=n_init
    )

    # Run the new pipeline:
    anubis.run()
    # All results -> 'outputs/expanding_eval_results.csv'
    # Models -> 'outputs/models'