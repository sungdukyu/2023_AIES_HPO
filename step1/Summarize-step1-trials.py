import os, glob
import pandas as pd
import numpy as np
from scipy import stats
import json
import pickle

RESULTS = {}

for kproj in ['P05', 'P25', 'P50', 'P100']:
    dir_results = './results'
    dir_trials = os.path.join(dir_results, kproj, 'trial_*')
    
    RESULTS[kproj] = {}
    for ktrial in glob.glob(dir_trials):
        trial_id = ktrial.split('/')[-1]
        RESULTS[kproj][trial_id] = {}
        
        # json
        f_json  = ktrial + '/trial.json'
        with open(f_json) as f:
          work = json.load(f)
        num_layers = work['hyperparameters']['values']['num_layers']
        
        # KerasTuner has a bug.
        # For beginning trials, trial.json record fails to record every hp values of units-per-layer.
        if len(work['hyperparameters']['values'])-1 < num_layers:
            print('ERROR (%s): num_layers=%d, num_units=%d'%(ktrial, num_layers, len(work['hyperparameters']['values'])-1))
            del RESULTS[kproj][trial_id]
            continue
        
        units = np.array( [work['hyperparameters']['values']['units_%d'%(k)] for k in range(num_layers)] )
        RESULTS[kproj][trial_id]['num_layers'] = num_layers
        RESULTS[kproj][trial_id]['units'] = units
        
        units_input  = 16
        units_output = 4
        units_all = np.array([units_input, *units, units_output])
        RESULTS[kproj][ktrial]['num_parameters'] = np.sum((units_all[:-1]+1) * units_all[1:])
        
        # csv
        f_csv = ktrial + '/trial_epoch_metrics_execution-01.csv'
        
        if os.path.getsize(f_csv)==0: 
            del RESULTS[kproj][trial_id] 
            continue
        
        work2 = pd.read_csv(f_csv)['val_loss']
        work2 = work2.rename(trial_id)
        RESULTS[kproj][trial_id]['val_loss'] = work2 
        RESULTS[kproj][trial_id]['epochs'] = len(work2)
        RESULTS[kproj][trial_id]['min_val_loss'] = work2.min()
        
with open("RESULTS.pkl","wb") as f:
    pickle.dump(RESULTS, f)
