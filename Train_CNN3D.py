#!/usr/bin/python3

"""
This script reads a train data set specified in a configuration file and trains a CNN.

"""

#nohup python3 Train_CNN3D.py 3dCNN_2Conv_2x2x2_2Dense_batch512_20epoES5 ./config/FF100/config_Train2dCNN_MuonID_FF100.yml &
#nohup python3 Train_CNN3D.py 3dCNN_2Conv_2x2x2_2Dense_batch512_20epoES5 &
#nohup python3 Train_CNN3D.py 3dCNN_2Conv_1x1x1_2Dense_batch128_20epoES5 &
#nohup python3 Train_CNN3D.py 3dCNN_LogMinMaxScaler_2Conv_2Dense_batch128_150epoES5 &
#nohup python3 Train_CNN3D.py 3dCNN_LogMinMaxScaler_Sigmoid_2Conv_2Dense_batch512_N20000 &
#nohup python3 Train_CNN3D.py 3dCNN_NoNorm_Sigmoid_2Conv_2Dense_batch512_N20000 &

#nohup python3 Train_CNN3D.py /home/cosmo/borjasg/projects/DataChallenges/MODE_2022/config/config_NoNorm_Simple.yml &
#nohup python3 Train_CNN3D.py /home/cosmo/borjasg/projects/DataChallenges/MODE_2022/config/config_NoNorm_Simple_complete.yml &
#nohup python3 Train_CNN3D.py ./config/config_NoNorm_5x2x1.yml &
#nohup python3 Train_CNN3D.py ./config/config_Norm_5x2x1.yml &
#nohup python3 Train_CNN3D.py ./config/config_Norm_Simple_complete.yml &
#nohup python3 Train_CNN3D.py ./config/config_NoNorm_5x2x1_LSTM.yml &





import time

# starting time
start = time.time()



import pandas as pd
import numpy as np
import os
import sys

import matplotlib.pyplot as plt
import matplotlib as mpl
mpl.style.use('~/.matplotlib/matplotlibrc.bin')
mpl.use('agg')


from sklearn.preprocessing import MinMaxScaler

import h5py
import tables

import tensorflow as tf
import keras.backend.tensorflow_backend as tfback
from keras.models import Sequential, Model
from keras.layers import Dense, Conv1D, Conv3D, Flatten, Reshape, concatenate, Input,LSTM,TransformerDecoder
from keras.optimizers import SGD,Adam
from keras.wrappers.scikit_learn import KerasRegressor
from keras.backend import one_hot
from keras.utils.np_utils import to_categorical
from keras.callbacks import EarlyStopping

from keras.models import load_model

from sklearn.metrics import accuracy_score,f1_score,precision_score,recall_score
from sklearn.metrics import r2_score,mean_absolute_error,mean_squared_error
import seaborn as sns



import joblib

import yaml
import random



def _get_available_gpus():  
    if tfback._LOCAL_DEVICES is None:  
        devices = tf.config.list_logical_devices()  
        tfback._LOCAL_DEVICES = [x.name for x in devices]  
    return [x for x in tfback._LOCAL_DEVICES if 'device:gpu' in x.lower()]


tfback._get_available_gpus = _get_available_gpus

if (len(sys.argv) < 2):
    print("ERROR. No arguments were given to identify the run", file = sys.stderr)
    print("Please indicate the ID for the experiment and the yaml config file")
    sys.exit(1)






with open(sys.argv[1], 'r') as ymlfile:   
    cfg = yaml.safe_load(ymlfile)


exp_ID = cfg["ML_Models"]["3dCNN_out_name"]
print("Experiment ID: ", exp_ID)
print("Exp. Config.: ", cfg)


random_st=0 
seed = random_st
np.random.seed(seed)
random.seed(seed)
#tf.random.set_seed(seed)


#############################################################################################

#######################
### Python Functions###
#######################

def LogMinMaxScaler(data):
    upper_limit = 0.2
    #Loop over events
    for i in range(len(data)):
        #Apply limits to data
        data[i][data[i]==0] = 10**(-5)
        data[i][data[i]>upper_limit] = upper_limit
        #Apply log scale to values different to 0
        data[i] = np.log10(data[i])
        #Scale from 1 to 10 values different to 0
        scaler = MinMaxScaler(feature_range=(0,10))
        scaler.fit(data[i].reshape(-1,1))
        data[i] = scaler.transform(data[i].reshape(-1,1)).reshape(data[i].shape)
        del scaler
    return data



def NormalizeAllData(x):
    for j in range(x.shape[0]):
        x[j,:] = x[j,:]/np.sum(x[j,:])
    return x

###############################################################################################



#Read Train data
DATA = '/lstore/lattes/borjasg/projects/DataChallenge_MODE2022/'
TrainDf = h5py.File(DATA+'train.h5', 'r')


#Try with a subset (there are 100k events)
n = cfg["global"]["Nevents"]
n_train = int(n*cfg["global"]["Train_prop"]) #Get 80% of the events for training

#Get train data set
X_train = TrainDf['x0'][:n_train]
Y_train = TrainDf['targs'][:n_train]
Y_train_NotReshape = TrainDf['targs'][:n_train]
Y_train_shape = Y_train.shape

Y_train = Y_train.reshape(Y_train.shape[0],-1)

#Get validation data set
X_val = TrainDf['x0'][n_train:n]
Y_val = TrainDf['targs'][n_train:n]
Y_val_NotReshape = TrainDf['targs'][n_train:n]
Y_val_shape = Y_val.shape
Y_val = Y_val.reshape(Y_val.shape[0],-1)


#Close h5 file
TrainDf.close()


#Get dimensions of the data
Npixels_x = 10
Npixels_y = 10
Npixels_z = 10


#############
#Build models 
#############


def build_model():
    model_m = Sequential()
    model_m.add(Conv3D(kernel_size = (5,2,1),strides=(2,1,1), filters = 25, activation='relu', input_shape=(Npixels_x,Npixels_y,Npixels_z,1)))
    #model_m.add(Conv3D(kernel_size = (1,1,1), filters = 10, activation='relu', input_shape=(Npixels_x,Npixels_y,Npixels_z,1)))
    model_m.add(Flatten())
    #model_m.add(Dense (30, activation='relu'))
    #model_m.add(Dense (50, activation='relu'))
    model_m.add(Dense (100, activation='relu'))
    model_m.add(Dense(1000, activation = 'sigmoid'))
    
    adam_optimizer = Adam(learning_rate=0.001, beta_1=0.9, beta_2=0.999, amsgrad=False)
    model_m.compile(loss='mean_squared_error', optimizer=adam_optimizer)
    return model_m



#Normalisation

Norm_opt = cfg["global"]["Normalisation"]

if Norm_opt=="LogMinMaxScaler":
    X_train = LogMinMaxScaler(X_train)
    X_val = LogMinMaxScaler(X_val)
elif Norm_opt=="NormSignal":
    X_train = NormalizeAllData(X_train)    
    X_val = NormalizeAllData(X_val)




#define model...
epochs = int(cfg["FE"]["Nepochs"])
batch = int(cfg["FE"]["batch"])
patience_ES = int(cfg["FE"]["EarlyStopping"])


estimator = KerasRegressor(build_fn=build_model, nb_epoch=epochs, verbose=1)
#Reshape to 3D channels:
X_train_rshp = X_train.reshape((X_train.shape[0],Npixels_x,Npixels_y,Npixels_z,1))

del X_train


X_val = X_val.reshape((X_val.shape[0],Npixels_x,Npixels_y,Npixels_z,1))


callback = EarlyStopping(monitor='val_loss', patience=patience_ES)

model_name = exp_ID+'_'+Norm_opt+'_batch'+str(batch)+'_'+str(epochs)+'epoES'+str(patience_ES)+"_N"+str(n)

output_model_name = cfg["storage"]["dataPath_models"]+'keras-'+model_name+'.h5'

estimator.fit(X_train_rshp,Y_train, batch_size = batch, epochs=epochs, validation_data=(X_val, Y_val), callbacks=[callback])
estimator.model.save(output_model_name)
#del X_train, Y_train, X_train_rshp, X_train_int


# end time
end = time.time()
time_program = np.round((end - start),2)

if (time_program<60): 
    print("Runtime of the program is ",time_program," s.")
elif (time_program>=60)*(time_program<3600): 
    time_program = np.round(time_program/60,2)
    print("Runtime of the program is ",time_program," mins.")
elif (time_program>=3600):
    time_program = np.round(time_program/3600,2)
    print("Runtime of the program is ",time_program," h.")
    
