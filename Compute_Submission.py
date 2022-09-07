#!/usr/bin/python3

"""
This script load a model and computes the submission file.

Example to run the program:

nohup python3 Compute_Submission.py ./config/config_NoNorm_5x2x1.yml &
"""




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



from keras.models import load_model


import seaborn as sns



import joblib

import yaml
import random


import tensorflow as tf
import keras.backend.tensorflow_backend as tfback
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
    #Loop over events
    for i in range(len(data)):
        #Apply log scale to values different to 0
        data[i][data[i]!=0] = np.log10(data[i][data[i]!=0])
        #Scale from 1 to 10 values different to 0
        scaler = MinMaxScaler(feature_range=(1,10))
        scaler.fit(data[i][data[i]!=0].reshape(-1,1))
        data[i][data[i]!=0] = scaler.transform(data[i][data[i]!=0].reshape(-1,1)).reshape(-1)
        del scaler
    return data

###############################################################################################



#Read Train data
DATA = '/lstore/lattes/borjasg/projects/DataChallenge_MODE2022/'
TrainDf = h5py.File(DATA+'train.h5', 'r')


#Try with a subset (there are 100k events)
n = cfg["global"]["Nevents"]
n_train = int(n*cfg["global"]["Train_prop"]) #Get 80% of the events for training


"""n = 20000
n_train = int(0.8*n)"""


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

#Get test data set
TestDf = h5py.File(DATA+'test.h5', 'r')

X_test = np.asarray(TestDf['x0'])

TestDf.close()



#Get dimensions of the data
Npixels_x = 10
Npixels_y = 10
Npixels_z = 10




#############
#Read model 
#############


#Normalisation

Norm_opt = "NoNorm"
if cfg["global"]["Normalisation"]:
    X_train = LogMinMaxScaler(X_train)
    X_val = LogMinMaxScaler(X_val)
    X_test = LogMinMaxScaler(X_test)
    Norm_opt = "LogMinMaxScaler"



X_test_shape = X_test.shape


#Reshape to 3D channels:
X_train = X_train.reshape((X_train.shape[0],Npixels_x,Npixels_y,Npixels_z,1))
X_val = X_val.reshape((X_val.shape[0],Npixels_x,Npixels_y,Npixels_z,1))
X_test = X_test.reshape((X_test.shape[0],Npixels_x,Npixels_y,Npixels_z,1))



model_name = cfg["ML_Models"]["Trained_model"]
output_model_name = cfg["storage"]["dataPath_models"]+model_name+'.h5'


estimator = load_model(output_model_name)

#Load thesholds
ModelFile = h5py.File(output_model_name, 'r')

opt_thresholds = np.asarray(ModelFile['opt_thresholds'])

ModelFile.close()


folder = "/home/cosmo/borjasg/projects/DataChallenges/MODE_2022/imgs/results_opt_submit/"+model_name+"/"

if not os.path.exists(folder): os.mkdir(folder)


###################################################################################################################################33


def binary_iou(preds:np.ndarray, targs:np.ndarray, as_mean:bool=True, smooth:float=1e-17) -> float:
    r'''
    Assumes that preds and targs have a batch dimension, and no class dimension, i.e. (batch, z, x, y)
    If the batch dimesion is missing, e.g. your are computing the IOU for a single sample,
    add it with e.g. preds[None], targs[None]
    '''
    preds = preds.reshape(len(preds), 1, -1).astype(bool)
    targs = targs.reshape(len(targs), 1, -1).astype(bool)
    inter = (preds*targs).sum(-1)
    union = (preds+targs).sum(-1)
    iou = ((inter+smooth)/(union+smooth))  # Small epsilon in case of zero union
    if as_mean: iou = iou.mean() 
    return iou


#Predict train and plot
pred_model = np.asarray(estimator.predict(X_train)).reshape(Y_train_shape)
pred_val_model = np.asarray(estimator.predict(X_val)).reshape(Y_val_shape)
pred_test_model = np.asarray(estimator.predict(X_test)).reshape(X_test_shape)


for layer in range(X_train.shape[1]):
    best_thresh = opt_thresholds[layer]
    pred_model[:,layer] = (pred_model[:,layer] > best_thresh).astype(int)
    pred_val_model[:,layer] = (pred_val_model[:,layer] > best_thresh).astype(int)
    pred_test_model[:,layer] = (pred_test_model[:,layer] > best_thresh).astype(int)

iou_train = binary_iou(pred_model, Y_train_NotReshape)
iou_val = binary_iou(pred_val_model, Y_val_NotReshape)



print("IOU (Train) = ",iou_train)
print("IOU (Validation) = ",iou_val)


output_folder = "/lstore/lattes/borjasg/projects/DataChallenge_MODE2022/submissions/"+exp_ID+"/"


if not os.path.exists(output_folder): os.mkdir(output_folder)



full_name = 'BORJA_GONZALEZ_ID01'
with h5py.File(output_folder+full_name+'_test_preds.h5', 'w') as h5:
    h5.create_dataset("preds", data=pred_test_model.astype("int"), dtype="int", compression='lzf')




def plot_x0_targ(x0:np.ndarray, targ:np.ndarray,plotname) -> None:
    n_layers = x0.shape[0]
    fig, axs = plt.subplots(n_layers,2, figsize=(15, 30))
    pred_cbar_ax = fig.add_axes([0.45, 0.25, 0.03, 0.5])
    true_cbar_ax = fig.add_axes([0.90, 0.25, 0.03, 0.5])
    for layer in range(n_layers-1,-1,-1):
        sns.heatmap(x0[layer], ax=axs[n_layers-layer-1,0], vmin=np.nanmin(x0), vmax=np.nanmax(x0), cbar=(layer==0), cbar_ax=pred_cbar_ax if layer == 0 else None, square=True, cmap='viridis')
        sns.heatmap(targ[layer], ax=axs[n_layers-layer-1,1], vmin=np.nanmin(targ), vmax=np.nanmax(targ), cbar=(layer==0), cbar_ax=true_cbar_ax if layer == 0 else None, square=True, cmap='viridis')
        axs[n_layers-layer-1][0].set_ylabel(f"Layer {layer}")
    axs[-1][0].set_xlabel("Prediction")
    axs[-1][1].set_xlabel("Target")
    fig.tight_layout()
    plt.savefig(plotname)
    plt.close()


idxs = np.arange(3)


for idx in idxs:
    name = folder+"Train_id"+str(idx)+"_"+model_name+".pdf"
    plot_x0_targ(pred_model[idx], Y_train_NotReshape[idx],plotname=name)
    name = folder+"Val_id"+str(idx)+"_"+model_name+".pdf"
    plot_x0_targ(pred_val_model[idx], Y_val_NotReshape[idx],plotname=name)





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
    
