import xarray as xr
import numpy as np
import pandas as pd
import tensorflow as tf
import os
import yaml

def read_yaml(file_path):
    with open(file_path, "r") as f:
        return yaml.safe_load(f)

def set_environment(num_gpus_per_node="8"):
    nodename = os.environ['SLURMD_NODENAME']
    procid = os.environ['SLURM_LOCALID']
    print(nodename)
    print(procid)
    stream = os.popen('scontrol show hostname $SLURM_NODELIST')
    output = stream.read()
    oracle = output.split("\n")[0]
    print(oracle)
    if procid==num_gpus_per_node:
        os.environ["KERASTUNER_TUNER_ID"] = "chief"
        os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    else:
        os.environ["KERASTUNER_TUNER_ID"] = "tuner-" + str(nodename) + "-" + str(procid) 
        os.environ["CUDA_VISIBLE_DEVICES"] = procid

    os.environ["KERASTUNER_ORACLE_IP"] = oracle + ".ib.bridges2.psc.edu" # Use full hostname
    os.environ["KERASTUNER_ORACLE_PORT"] = "8000"
    print("KERASTUNER_TUNER_ID:    %s"%os.environ["KERASTUNER_TUNER_ID"])
    print("KERASTUNER_ORACLE_IP:   %s"%os.environ["KERASTUNER_ORACLE_IP"])
    print("KERASTUNER_ORACLE_PORT: %s"%os.environ["KERASTUNER_ORACLE_PORT"])
    #print(os.environ)

def vectorize(ds:xr.Dataset, v:list):
    for k, kvar in enumerate(v):
        if k == 0:
            train_input_vectorized = ds[kvar].to_pandas()
        else:
            train_input_vectorized = pd.concat([train_input_vectorized, ds[kvar].to_pandas()], axis=1)
            
    return train_input_vectorized.values

def main():
    # EAGLES variables:
    # input_vars: ['tair', 'pressure', 'rh', 'wbar', 'sigmag', 'num_aer', 'r_aer', 'kappa']
    # output_vars: ['fn', 'fm', 'fluxm', 'fluxn', 'smax']
    # extra_output_vars: ['bad_samples', 'timed_out', 'super_cooled']

    #0 Load CONFIG File
    CONFIG = read_yaml("./config/Task01_Run01_data-volume-p01.yml")
    projname = "%s%.2f"%(CONFIG['project']['prefix'], CONFIG['dataset']['volume_factor_effective'])
    print("CONFIG File: %s"%CONFIG)
    print("Project Name:: %s"%projname)

    #1 Load
    f_orig = xr.open_dataset(CONFIG['dataset']['file_path']) #, chunks={'nsamples':1048576}) # just hardwire

    # Cleaning bad samples
    # (The input data is already cleaned, so not cleaning here again)
    if False:
        # check NaNs
        for kvar in list(f_orig.variables):
            n_nans = np.sum(np.isnan(f_orig[kvar].values))
            print('%s: %d (%.5f)'%(kvar, n_nans, n_nans/f_orig[kvar].size))
        
        # check bad samples (-987654{,3,32} is a flag for bad samples)
        print((np.sum(f_orig['bad_samples'].values == -987654), np.sum(f_orig['timed_out'].values == -9876543),  np.sum(f_orig['super_cooled'].values == -98765432)))

        # remove nans and bad_samples
        ind_keep = ~( (np.isnan(f_orig['tair'].values))\
                    +(f_orig['bad_samples'].values == -987654)\
                    +(f_orig['timed_out'].values == -9876543)\
                    +(f_orig['super_cooled'].values == -98765432))
        f_orig=f_orig.isel(nsamples=ind_keep)
        
    #2 Select variables relevant for emulator 
    f_train  = f_orig[[ *CONFIG['variables']['input'], *CONFIG['variables']['output'] ]]
    f_train0 = f_orig[[ *CONFIG['variables']['input'], *CONFIG['variables']['output'] ]] # copy

    #5 Normalize
    mu    = f_train.mean('nsamples')
    sigma = f_train.std('nsamples')
    f_train = f_train - mu
    f_train = f_train / sigma

    #6 Vectorize input / output
    # (f_train is normalized, but f_train0 is not)
    input_vars =  CONFIG['variables']['input']
    output_vars = CONFIG['variables']['output']
    train_input_vectorized  = vectorize(f_train,  input_vars)  # using f_train
    train_output_vectorized = vectorize(f_train0, output_vars) # using f_train0

    #6.b divide train into train vs validation
    split = CONFIG['dataset']['validation_split']
    n_train = train_input_vectorized.shape[0]
    n_val = int(n_train*split)
    x_val = train_input_vectorized[:n_val]
    x_train = train_input_vectorized[n_val:]
    y_val = train_output_vectorized[:n_val]
    y_train = train_output_vectorized[n_val:]

    # training
    from keras import models
    from keras import layers
    from keras import callbacks
    import keras_tuner as kt

    # hypermodel
    def build_model(hp):
        model = models.Sequential()

        for i in range(hp.Int("num_layers", 1, 20)):
            model.add(
                layers.Dense(
                    # Tune number of units separately.
                    units=hp.Choice(f"units_{i}", [8, 16, 32, 64, 128, 256, 512, 1024, 2048]),
                    activation="relu",
                )
            )

        model.add(layers.Dense(train_output_vectorized.shape[1], activation='sigmoid'))
        model.compile(**CONFIG['compile'])

        return model

    # search set up
    tuner = kt.RandomSearch(
                            build_model,
                            project_name = projname,
                            **CONFIG['RandomSearch']
                           )
    tuner.search_space_summary()

    # search
    tuner.search(x_train, y_train,
                 validation_data=(x_val, y_val),
                 **CONFIG['search'],
                 callbacks=[callbacks.EarlyStopping('val_loss', patience=5)] # CSVLogger callback is also included by the direct src mod.
                 )

if __name__ == '__main__':
    # setting env variables for distributed search
    set_environment()

    # main HPO code
    main()
