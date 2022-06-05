#!/usr/bin/env python
# coding: utf-8

# # Part 1: Sun Centric Planet to Planet Solution

import warnings
warnings.filterwarnings("ignore", message=r"Passing", category=FutureWarning)
warnings.filterwarnings("ignore", message=r"calling", category=FutureWarning)

import math 


import tensorflow.keras.backend as K
from tensorflow.keras.models import load_model  
import wandb
import tensorflow as tf
from icecream import ic
from numpy import savetxt
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from GenerateDataSet import ReturnDataSet
from DNN_tools import TrainNew, TrainOld, ComparePerformance, WandB_Sweep



import sys

print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))

# total arguments
n = len(sys.argv)
print("Total arguments passed:", n)
 
# Arguments passed
print("\nName of Python script:", sys.argv[0])
 
print("\nArguments passed:", end = " ")
for i in range(1, n):
    print(sys.argv[i], end = " ")

print("\n")

RunType = sys.argv[1]
if RunType =='sweep':
  num_runs = int(sys.argv[2])

# NN Input and Ouput

x_train, y_train, x_test, y_test =  ReturnDataSet(OldData=False, DataSetSize=10000, format='kep', Normalise=True, ReturnDF=False )

#savetxt('./Lambert_Solutions/Lambert_Solutions_x', np.vstack([x_train, x_test]), delimiter = ',')
#savetxt('./Lambert_Solutions/Lambert_Solutions_y', np.vstack([y_train, y_test]), delimiter = ',')


# In[5]:


ic(x_test[0])
ic(x_train[0])

ic(y_train[0])
ic(y_test[0])



# # Neural Network Training from Scratch
# 

# In[11]:


def config_generator(optimizer, batch_size, lr, Layer_Units, epochs):
    config = {}
    config['optimizer']    = optimizer
    config['batch_size']   = batch_size
    config['learning rate']= lr
    config['Layer_Units']  = Layer_Units
    config['epochs']       = epochs
    return config

#config1  
config1 = config_generator(optimizer='Adamax',
                            batch_size=3010,    
                            lr=0.01635, 
                            Layer_Units= [50, 50, 50, 50, 50, 50, 50, 50],
                            epochs=10000)

# config1 = config_generator(optimizer='Adamax',
#                             batch_size=3010,    
#                             lr=0.01635, 
#                             Layer_Units= [1000, 1000],
#                             epochs=10000)


config2 = config_generator(optimizer='Adamax',
                            batch_size=3010,
                            lr=0.01635, 
                            Layer_Units= [200,200,200,200,200],
                            epochs=10000)


configs = [config1, config2]


# In[12]:

if RunType=='single':
  for config in configs:
        tf.keras.backend.clear_session()

        callback_config = {}
        callback_config['TB'] = {}
        callback_config['EarlyStopping'] = {}
        callback_config['SaveBest'] = {}
        callback_config['Val Loss Threshold'] = {}
        callback_config['Reduce lr'] = {}
        callback_config['wandb'] = {}
        callback_config['Print_nEpochs'] = {}
        
        callback_config['TB']["include"] = False

        callback_config['EarlyStopping']["include"] = True
        if callback_config['EarlyStopping']["include"]:
            callback_config['EarlyStopping']["min delta"] = .5
            callback_config['EarlyStopping']["patience"] = 5000

        callback_config['SaveBest']["include"] = False

        callback_config['Val Loss Threshold']["include"] = True
        if callback_config['Val Loss Threshold']["include"]:
            callback_config['Val Loss Threshold']["Threshold"] = 0.20

        callback_config['Reduce lr']["include"] = True
        if callback_config['Reduce lr']["include"]:
            callback_config['Reduce lr']["factor"] = 0.8
            callback_config['Reduce lr']["patience"] = 500
            callback_config['Reduce lr']["min delta"] = 0.1
            callback_config['Reduce lr']["min lr"] = 0.000001


        callback_config['Print_nEpochs']["include"] = True
        if callback_config['Print_nEpochs']["include"]:
            callback_config['Print_nEpochs']["fequency"] = 1000


        callback_config['wandb']["include"] = True
        

        NewModel = TrainNew(x_train, y_train, x_test, y_test, config=config)

        NewModel.Build(Inputs=tf.keras.Input(x_train.shape[1]), Dropout=[0,0], BatchNorm = False)

        NewModel.Compile(loss_func = "mape")

        Model, History = NewModel.Train(callback_config=callback_config, verbose=0)
        #vars(NewModel)
        if History.history['val_loss'][-1]<25: 
            Model.save(r'./saved_models/BestModel.h5')
                

        example = 300
        ic(np.mean(abs((Model.predict(x_test)-y_test)/y_test, axis=0)))
# Model.save(r'./saved_models/BestModel.h5')



# In[8]:

'''
config1 = config_generator(optimizer='Adam',
                            batch_size=5000,    
                            lr=0.01, 
                            Layer_Units= [256, 64, 16, 8],
                            epochs=5000)

config2 = config_generator(optimizer='Adamax',
                            batch_size=3000,
                            lr=0.01, 
                            Layer_Units= [50,50,50,50],
                            epochs=15000)


configs = []


for config in configs:
            #tf.keras.backend.clear_session()

            callback_config = {}
            callback_config['TB'] = {}
            callback_config['EarlyStopping'] = {}
            callback_config['SaveBest'] = {}
            callback_config['Val Loss Threshold'] = {}
            callback_config['Reduce lr'] = {}
            callback_config['wandb'] = {}
            callback_config['Print_nEpochs'] = {}
            
            callback_config['TB']["include"] = False

            callback_config['EarlyStopping']["include"] = False
            callback_config['EarlyStopping']["min delta"] = 0.1
            callback_config['EarlyStopping']["patience"] = 1000

            callback_config['SaveBest']["include"] = False

            callback_config['Val Loss Threshold']["include"] = True
            callback_config['Val Loss Threshold']["Threshold"] = 0.20

            callback_config['Reduce lr']["include"] = True
            callback_config['Reduce lr']["factor"] = 0.8
            callback_config['Reduce lr']["patience"] = 500
            callback_config['Reduce lr']["min delta"] = 0.1
            callback_config['Reduce lr']["min lr"] = 0.00001

            callback_config['Print_nEpochs']["include"] = True
            callback_config['Print_nEpochs']["fequency"] = 100


            callback_config['wandb']["include"] = True

            OldModel = load_model('saved_models/Bestmodel.h5')
            NewModel = TrainOld(OldModel, x_train, y_train, x_test, y_test, config=config)
            NewModel.Compile(loss_func = "mape")
            NewModel.Train(callback_config=callback_config, verbose=0)

            wandb.finish()
'''


# In[24]:


# test_model =  load_model('saved_models/Bestmodel.h5')


# In[ ]:


# ComparePerformance().Compare()


# # WandB Sweep Setup

# In[16]:


sweep_config = {
  "name" : "my-sweep",
  "method" : "bayes",
  "metric" : {
                    "name": "val_loss",
                    "goal"   : "minimize"
  },
  "parameters" : {
                    'optimizer': {
                      'values': ['Adamax']
                    },
                    "epochs" : {
                      'values': [50000]
                    },
                    "learning rate" :{
                      "distribution": "log_uniform_values",
                      "min": 0.01,
                      "max": 0.5
                    },
                    "batch_size":{
                      "distribution": "int_uniform",
                      "min" : 1000,
                      "max" : 10000
                    },
                    "Layer_Units":{
                      "values" : [[10000], [50,50,50], [50,50,50,50], [50,50,50,50,50], [300,300,300,300]]
                    },
                    "loss_func":{
                      "values": ["npe"]
                    }
                  }
                }


# In[17]:


if 0: 
    Model = load_model('saved_models/Further_Trained20211121-201410-Adam-22k-Params-500-Epochs-1000')
else:
    Model = None

if RunType=='sweep':
  print('Sweeping')
  sweep = WandB_Sweep(sweep_config, x_train, y_train, x_test, y_test)

  sweep.Initialise_WandB_Sweep(FurtherTrain=False, ModelName = 'Further_Trained20211121-201851-Adamax-22k-Params-910-Epochs-1828')
  sweep.Run_WandB_Sweep(count=num_runs)



# 
