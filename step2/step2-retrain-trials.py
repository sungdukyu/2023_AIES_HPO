import xarray as xr
import numpy as np
import pandas as pd
import os
import sys
import json
import tensorflow as tf
from keras import models
from keras import layers
from keras import callbacks

### USER parameters ###
f_json = sys.argv[1] # file that contains KerasTunter trial info
                     # (eg) ../step1/results/P05/trial_00044/trial.json
f_trained_model = sys.argv[2] # file name to save trained model
                              # (eg) ./top_models/P05-trial_00044.hdf5
f_dataset = '~/data/all_samples_output_cleaned_shuffled.nc'
vars_input  = ['tair', 'pressure', 'rh', 'wbar', 'num_aer', 'r_aer', 'kappa']
vars_output = ['fn']
validation_split = 0.2
compile_opt = {'optimizer': 'Adam',
               'loss': 'mse',
               'metrics': ['mae']
              }
fit_opt = {'batch_size': 1024,
           'epochs': 200
          }
####

print('f_json: ', f_json)
print('f_trained_model: ', f_trained_model)

def vectorize(ds:xr.Dataset, v:list):
    for k, kvar in enumerate(v):
        if k == 0:
            train_input_vectorized = ds[kvar].to_pandas()
        else:
            train_input_vectorized = pd.concat([train_input_vectorized, ds[kvar].to_pandas()], axis=1)
            
    return train_input_vectorized.values

def main():
    #1 Load training dataset
    f_orig = xr.open_dataset(f_dataset)

    #2 Select variables relevant for emulator 
    f_train  = f_orig[vars_input]  # for input  -> gonna be normalized
    f_train0 = f_orig[vars_output] # for output -> not

    #3 Normalize
    mu    = f_train.mean('nsamples')
    sigma = f_train.std('nsamples')
    f_train = f_train - mu
    f_train = f_train / sigma

    #4 Vectorize input / output
    # (f_train is normalized, but f_train0 is not)
    train_input_vectorized  = vectorize(f_train,  vars_input)  # using f_train
    train_output_vectorized = vectorize(f_train0, vars_output) # using f_train0

    #5 divide train into train vs validation
    n_train = train_input_vectorized.shape[0]
    n_val   = int(n_train * validation_split)
    x_val   = train_input_vectorized[:n_val]
    x_train = train_input_vectorized[n_val:]
    y_val   = train_output_vectorized[:n_val]
    y_train = train_output_vectorized[n_val:]

    #6 get trial information
    with open(f_json) as f:
        work = json.load(f)
    num_layers = work['hyperparameters']['values']['num_layers']
    if len(work['hyperparameters']['values'])-1 < num_layers:
        raise Exception('ERROR: num_layers=%d, num_units=%d'%(num_layers, len(work['hyperparameters']['values'])-1))
    else:
        units = np.array( [work['hyperparameters']['values']['units_%d'%(k)] for k in range(num_layers)] )

    #7 build model
    def build_model_hp(units: np.array):
        model = models.Sequential()
        input_length  = train_input_vectorized.shape[1]
        output_length = train_output_vectorized.shape[1]

        for k in units:
            model.add(layers.Dense(k,
                                   activation='relu',
                                   input_shape=(input_length,) # ignored after the 1st layer
                                  ))

        model.add(layers.Dense(output_length,
                               activation='sigmoid',
                              ))

        model.compile(**compile_opt)
        return model
    model = build_model_hp(units)
    model.summary()

    #8 train
    checkpoint = callbacks.ModelCheckpoint(filepath=f_trained_model,
                                           save_weights_only=False,
                                           verbose=1,
                                           monitor='val_loss',
                                           save_best_only=True)
    csv_logger = callbacks.CSVLogger(filename=f_trained_model+'.epoch_metrics.csv',
                                     append=False)
    history = model.fit(x_train, y_train,
                        validation_data=(x_val, y_val),
                        callbacks=[checkpoint, csv_logger],
                        verbose=2,
                        **fit_opt
                       )


if __name__ == '__main__':
    # limit memory preallocation
    physical_devices = tf.config.list_physical_devices('GPU')
    print(physical_devices)
    tf.config.experimental.set_memory_growth(physical_devices[0], True) # only using a single GPU

    # main HPO code
    main()
