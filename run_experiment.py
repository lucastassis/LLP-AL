import sys
sys.path.append("src/")
import numpy as np
import pandas as pd
import time
import glob
from engine_tabular import run_kld_entr_sampling, run_mlp, run_dllp, run_entr_sampling, run_rand_sampling, run_active_mlp
from sklearn.model_selection import train_test_split
import warnings
import argparse
warnings.simplefilter(action='ignore', category=FutureWarning)

parser = argparse.ArgumentParser()
parser.add_argument("--dataset", "-d", required=True)
parser.add_argument("--budget", "-b", required=True, type=int)
parser.add_argument("--batchsize", "-s", required=True, type=int)
parser.add_argument("--outputfile", "-o", required=True)

args = parser.parse_args()

n_reveal = np.arange(args.batchsize, args.budget + args.batchsize, args.batchsize)
seeds = [42, 1717, 4343, 301293, 430319]

data = pd.read_parquet(args.dataset)
data = data.rename(columns={"y" : "target"})
results_df = pd.DataFrame(columns=["name", "lambda", "fold", "num_reveal", "acc", "pr", "re", "f1"])
print(f"INFO: training on dataset {args.dataset}")
for fold in range(0, 5):
    print(f"TRAIN UPDATE: start training and evaluating on fold {fold}")
    # split data
    train_df, test_df = train_test_split(data, test_size=0.2, random_state=seeds[fold])

    start = time.time()
    # mlp
    metrics = run_mlp(train_df=train_df, test_df=test_df)
    for n in n_reveal:
        new_results = {"name" : "SUP", "lambda" : np.nan, "fold" : fold, "num_reveal" : n, "acc" : metrics[0], "pr" : metrics[1], "re" : metrics[2], "f1" : metrics[3]}
        results_df = pd.concat([results_df, pd.DataFrame([new_results])], axis=0)
        
    # active mlp
    metrics = run_active_mlp(train_df=train_df, test_df=test_df, n_reveal=n_reveal, sample_batch_size=args.batchsize)
    for n, arg in zip(n_reveal, metrics):
        new_results = {"name" : "AL", "lambda" : np.nan, "fold" : fold, "num_reveal" : n, "acc" : arg[0], "pr" : arg[1], "re" : arg[2], "f1" : arg[3]}
        results_df = pd.concat([results_df, pd.DataFrame([new_results])], axis=0)

    # dllp
    metrics = run_dllp(train_df=train_df, test_df=test_df)
    for n in n_reveal:
        new_results = {"name" : "DLLP", "lambda" : np.nan, "fold" : fold, "num_reveal" : n, "acc" : metrics[0], "pr" : metrics[1], "re" : metrics[2], "f1" : metrics[3]}
        results_df = pd.concat([results_df, pd.DataFrame([new_results])], axis=0)

    for _lambda in [1e-3, 1e-2, 1e-1, 1e0, 1e1, 1e2, 1e3]:
        # rand dllp
        for n in n_reveal:
            metrics = run_rand_sampling(train_df=train_df, test_df=test_df, n_reveal=n, c=_lambda)
            new_results = {"name" : "RAND", "lambda" : _lambda, "fold" : fold, "num_reveal" : n, "acc" : metrics[0], "pr" : metrics[1], "re" : metrics[2], "f1" : metrics[3]}
            results_df = pd.concat([results_df, pd.DataFrame([new_results])], axis=0)
                
        # entr dllp
        metrics = run_entr_sampling(train_df=train_df, test_df=test_df, n_reveal=n_reveal, c=_lambda, sample_batch_size=args.batchsize)
        for n, arg in zip(n_reveal, metrics):
            new_results = {"name" : f"ENTR-BATCH={args.batchsize}", "lambda" : _lambda, "fold" : fold, "num_reveal" : n, "acc" : arg[0], "pr" : arg[1], "re" : arg[2], "f1" : arg[3]}
            results_df = pd.concat([results_df, pd.DataFrame([new_results])], axis=0)
                
        # entr dllp
        metrics = run_kld_entr_sampling(train_df=train_df, test_df=test_df, n_reveal=n_reveal, c=_lambda, sample_batch_size=args.batchsize)
        for n, arg in zip(n_reveal, metrics):
            new_results = {"name" : f"KLD-ENTR-BATCH={args.batchsize}", "lambda" : _lambda, "fold" : fold, "num_reveal" : n, "acc" : arg[0], "pr" : arg[1], "re" : arg[2], "f1" : arg[3]}
            results_df = pd.concat([results_df, pd.DataFrame([new_results])], axis=0)    
    print(f"TRAIN UPDATE: finished training and evaluating on fold {fold} in {time.time() - start}s")
results_df.to_csv(args.outputfile)
print(f"INFO: finished experiment on dataset {args.dataset}")
