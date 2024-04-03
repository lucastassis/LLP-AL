# On Combining Learning from Label Proportions and Active Learning in Low-Budget Regimes

Repository containing the code used in the work "On Combining Learning from Label Proportions and Active Learning in Low-Budget Regimes"

## Requirements
All the codes were written using Python3. You can install the requirements file by using:

```
pip install -r requirements.txt
```

Or install via [Anaconda](https://conda.io/projects/conda/en/latest/user-guide/getting-started.html) if you prefer =)

## Usage Minimal Example
To run a single experiment you have to provide: 
- the dataset (a dummy sample dataset is available on the folder `data/`); 
- the budget for the active learning setting; 
- the batch size for the active learning sampling; 
- and the output file name. 

For example, to run the experiments on 5 folds with budget 100, and batch size 10, you should run the following:

```
python run_experiment.py -d data/blobs.parquet -b 100 -s 10 -o results_blobs.csv
```

The results of all the models (SUP, AL, DLLP, RAND, ENTR, and KLD) for each fold and hyperparameter $\lambda$ will be saved on the file `results_blobs.csv`.
