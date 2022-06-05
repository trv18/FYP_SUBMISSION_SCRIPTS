from imp import load_dynamic
import tensorflow as tf
import tensorflow.keras.backend as K
from tensorflow.python.ops.gen_math_ops import Sum
import os
import wandb
import datetime
import numpy as np
import pprint


from numpy import loadtxt
from wandb.keras import WandbCallback
from tensorflow.keras.models import load_model
from icecream import ic
from tf_tools import CountParams

#------------------------------------------------------------------------
#------------------------------------------------------------------------
#------------------------------------------------------------------------
class TrainNew():
    def __init__(self, x_train, y_train, x_test, y_test, config):
        self.x_train = x_train
        self.y_train = y_train
        self.x_test = x_test
        self.y_test = y_test

        self.config = config

        self.optimizer   = config['optimizer']
        self.batch_size  = int(config['batch_size'])
        self.lr          = config['learning rate']
        self.Layer_Units = config['Layer_Units']
        self.epochs      = int(config['epochs'])

        ic(self.optimizer, self.batch_size, self.Layer_Units)       
        ic(int(np.floor(CountParams(x_train.shape[1],self.Layer_Units, y_train.shape[1])/1000)))
        TotalParams = int(np.floor(CountParams(x_train.shape[1], self.Layer_Units, y_train.shape[1])/1000))

            

        execution_time =  datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        self.Param_name = str(execution_time) + "-" + str(self.optimizer) + "-" +str(TotalParams) + "k-Params-" + str(self.epochs) + "-Epochs-" + str(self.batch_size)

    #------------------------------------------------------------------------
    #------------------------------------------------------------------------

    def Build(self, Inputs, BatchNorm=False, Regularisation=None, Dropout=[0.0,0.0]):
        self.Inputs = Inputs
        self.Layers = len(self.Layer_Units)
        self.Layer_Units = self.Layer_Units
        self.BatchNorm = BatchNorm
        self.Regularisation = Regularisation
        self.Dropout = Dropout

        x1 = tf.keras.layers.BatchNormalization()(self.Inputs)
        x1 = tf.keras.layers.Dense(units=self.Layer_Units[0],
                                    activation='relu',
                                                        )(x1)       

        x1 = tf.keras.layers.Dropout(self.Dropout[0])(x1)
        if self.BatchNorm:
                x1 = tf.keras.layers.BatchNormalization()(x1)

        for _layer in range(1,self.Layers):
            x1 = tf.keras.layers.Dense(units=self.Layer_Units[_layer])(x1)
            
            if self.BatchNorm:
                x1 = tf.keras.layers.BatchNormalization()(x1)
            x1 = tf.keras.layers.Activation(activation='relu')(x1)
            x1 = tf.keras.layers.Dropout(self.Dropout[1])(x1)

            

        Output = tf.keras.layers.Dense(units=3, name='velocities')(x1)
        
        self.Model = tf.keras.Model(inputs=self.Inputs, outputs=Output)
        self.Model.summary()

        return self.Model

    #------------------------------------------------------------------------
    #------------------------------------------------------------------------

    def Compile(self, loss_func = "mape"):

        # Specify the optimizer, and compile the model with loss functions for both outputs
        def SumPError(y_true, y_pred):
            return K.sum((K.abs((y_true-y_pred)/y_true*100)),axis=1, keepdims=True)

        def NormError(y_true, y_pred):
            return K.sqrt(K.sum((K.square((y_true-y_pred)/y_true*100)),axis=1, keepdims=True))

        optimizer = getattr(tf.keras.optimizers, self.optimizer)(learning_rate=self.lr)
        if loss_func == "spe": 
            loss_func = SumPError
        elif loss_func == 'npe':
            loss_func = NormError

        self.Model.compile(optimizer=optimizer,
                    loss={'velocities': loss_func},  
                    metrics={'velocities': [tf.keras.metrics.RootMeanSquaredError(name='rmse')]})
        
    #------------------------------------------------------------------------
    #------------------------------------------------------------------------

    def Train(self, callback_config=None, sweeping=False, verbose=0):

        self.callbacks =  []

        if callback_config['TB']["include"]: 
            #callback = TqdmCallback(verbose=1)
            logdir = os.path.join(r"C:\Users\vdwti\OneDrive - Imperial College London\YEAR 4\FYP\Scripts\Personal\logs", self.Param_name)
            TB_callback = tf.keras.callbacks.TensorBoard(logdir, histogram_freq=2)

            self.callbacks.append(TB_callback)

        #------------------------------------------------------------------------

        if callback_config['EarlyStopping']["include"]:
            model_checkpoint_callback = tf.keras.callbacks.EarlyStopping(monitor="loss",
                                                                        min_delta= callback_config['EarlyStopping']["min delta"],
                                                                        patience=callback_config['EarlyStopping']["patience"],
                                                                        verbose=2,
                                                                        mode="min",
                                                                        baseline=None,
                                                                        )

            self.callbacks.append(model_checkpoint_callback)

        #------------------------------------------------------------------------

        if callback_config['SaveBest']["include"]: 
            savebest_callback = tf.keras.callbacks.ModelCheckpoint(filepath=r'./saved_models/' + str(self.Param_name), mode='min', monitor='val_loss', verbose=2, save_best_only=True)
            
            self.callbacks.append(savebest_callback)

        #------------------------------------------------------------------------

        if callback_config['Val Loss Threshold']["include"]: 
            class MyCallBack(tf.keras.callbacks.Callback):
                        def on_epoch_end(self, epoch, logs={}):
                            if(logs.get("val_loss")< callback_config['Val Loss Threshold']["Threshold"] ): # you can change the value
                                self.model.stop_training=True

            val_loss_callback = MyCallBack()
            
            self.callbacks.append(val_loss_callback)

        #------------------------------------------------------------------------

        if callback_config['Reduce lr']["include"]: 
            reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(monitor='loss', 
                                                                factor=callback_config['Reduce lr']["factor"], mode='min', verbose=1,
                                                                min_delta=callback_config['Reduce lr']["min delta"],
                                                                patience=callback_config['Reduce lr']["patience"] ,
                                                                min_lr=callback_config['Reduce lr']["min lr"])
            
            self.callbacks.append(reduce_lr)

        #------------------------------------------------------------------------

        if callback_config['wandb']["include"]:
            if sweeping:
                pass
            else:
                wandb.init(project="SSH_Runs", entity="trv18", config=self.config)
               
            wandb.config.update(callback_config)
            self.callbacks.append(WandbCallback(monitor="val_loss"))
            ic(wandb.config)

        #------------------------------------------------------------------------

        if callback_config['Print_nEpochs']["include"]:
            class Print_nEpochs(tf.keras.callbacks.Callback):
                def on_epoch_end(self2, epoch, logs={}):
                    if epoch % callback_config['Print_nEpochs']["fequency"] == 0:
                        print(f'Epoch {epoch}/{self.epochs}')
                        print(f'Loss: {logs.get("loss")} - val_loss: {logs.get("val_loss")} \n' )

            self.callbacks.append(Print_nEpochs())

        #------------------------------------------------------------------------
        

        history = self.Model.fit(self.x_train, self.y_train,
                            epochs=self.epochs, 
                            batch_size=self.batch_size, 
                            verbose=verbose,  # 0 = blank, 1 = update per step, 2= update per epoch
                            callbacks=self.callbacks, 
                            validation_data=(self.x_test, self.y_test)
                            )

        return self.Model, history
        wandb.finish()

        #------------------------------------------------------------------------
        #------------------------------------------------------------------------

#------------------------------------------------------------------------
#------------------------------------------------------------------------   
#------------------------------------------------------------------------   

class TrainOld():
    def __init__(self, Model, x_train, y_train, x_test, y_test, config):
        self.x_train = x_train
        self.y_train = y_train
        self.x_test = x_test
        self.y_test = y_test

        self.config = config
        self.Model = Model

        self.optimizer   = config['optimizer']
        self.batch_size  = int(config['batch_size'])
        self.lr          = config['learning rate']
        self.Layer_Units = config['Layer_Units']
        self.epochs      = int(config['epochs'])

        ic(self.optimizer, self.batch_size, self.Layer_Units)

        TotalParams = int(np.floor(CountParams(x_train.shape[1],self.Layer_Units, y_train.shape[1])/1000))
        execution_time =  datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        self.Param_name = str(execution_time) + "-" + str(self.optimizer) + "-" +str(TotalParams) + "k-Params-" + str(self.epochs) + "-Epochs-" + str(self.batch_size)

    #------------------------------------------------------------------------
    #------------------------------------------------------------------------

    def Compile(self, loss_func = "mape"):

        # Specify the optimizer, and compile the model with loss functions for both outputs
        def SumPError(y_true, y_pred):
            return K.sum((K.abs((y_true-y_pred)/y_true*100)),axis=1, keepdims=True)

        def NormError(y_true, y_pred):
            return K.sqrt(K.sum((K.square((y_true-y_pred)/y_true*100)),axis=1, keepdims=True))

        optimizer = getattr(tf.keras.optimizers, self.optimizer)(learning_rate=self.lr)
        if loss_func == "spe": 
            loss_func = SumPError
        elif loss_func == 'npe':
            loss_func = NormError

        self.Model.compile(optimizer=optimizer,
                    loss={'velocities': loss_func},  
                    metrics={'velocities': [tf.keras.metrics.RootMeanSquaredError(name='rmse')]})
        
    #------------------------------------------------------------------------
    #------------------------------------------------------------------------

    def Train(self, callback_config=None,  sweeping=False, verbose=0):

        self.callbacks =  []

        if callback_config['EarlyStopping']["include"]:
            model_checkpoint_callback = tf.keras.callbacks.EarlyStopping(monitor="loss",
                                                                        min_delta= callback_config['EarlyStopping']["min delta"],
                                                                        patience=callback_config['EarlyStopping']["patience"],
                                                                        verbose=2,
                                                                        mode="min",
                                                                        baseline=None,
                                                                        )

            self.callbacks.append(model_checkpoint_callback)

        #------------------------------------------------------------------------

        if callback_config['SaveBest']["include"]: 
            savebest_callback = tf.keras.callbacks.ModelCheckpoint(filepath=r'./saved_models/' + str(self.Param_name), mode='min', monitor='val_loss', verbose=2, save_best_only=True)
            
            self.callbacks.append(savebest_callback)

        #------------------------------------------------------------------------

        if callback_config['Val Loss Threshold']["include"]: 
            class MyCallBack(tf.keras.callbacks.Callback):
                        def on_epoch_end(self, epoch, logs={}):
                            if(logs.get("val_loss")< callback_config['Val Loss Threshold']["Threshold"] ): # you can change the value
                                self.model.stop_training=True

            val_loss_callback = MyCallBack()
            
            self.callbacks.append(val_loss_callback)

        #------------------------------------------------------------------------

        if callback_config['Reduce lr']["include"]: 
            reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(monitor='loss', 
                                                                factor=callback_config['Reduce lr']["factor"], mode='min', verbose=1,
                                                                min_delta=callback_config['Reduce lr']["min delta"],
                                                                patience=callback_config['Reduce lr']["patience"] ,
                                                                min_lr=callback_config['Reduce lr']["min lr"])
            
            self.callbacks.append(reduce_lr)

        #------------------------------------------------------------------------

        if callback_config['wandb']["include"]:
            if sweeping:
                pass
            else:
                wandb.init(project="SSH_Runs", entity="trv18", config=self.config)
                wandb.config.update(callback_config)

            self.callbacks.append(WandbCallback(monitor="val_loss"))
            wandb.config = self.config

        #------------------------------------------------------------------------

        if callback_config['Print_nEpochs']["include"]:
            class Print_nEpochs(tf.keras.callbacks.Callback):
                def on_epoch_end(self2, epoch, logs={}):
                    if epoch % callback_config['Print_nEpochs']["fequency"] == 0:
                        print(f'Epoch {epoch}/{self.epochs} \n')
                        print(f'Loss: {logs.get("loss")} - val_loss: {logs.get("val_loss")}' )

            self.callbacks.append(Print_nEpochs())

        #------------------------------------------------------------------------
        
        history = self.Model.fit(self.x_train, self.y_train,
                            epochs=self.epochs, 
                            batch_size=self.batch_size, 
                            verbose=verbose,  # 0 = blank, 1 = update per step, 2= update per epoch
                            callbacks=self.callbacks, 
                            validation_data=(self.x_test, self.y_test)
                            )

        return self.Model, history


#------------------------------------------------------------------------
#------------------------------------------------------------------------
#------------------------------------------------------------------------


class WandB_Sweep:
    def __init__(self, sweep_config, x_train, y_train, x_test, y_test):
        self.sweep_config = sweep_config
        self.sweep_id = wandb.sweep(sweep_config, project="SweepDir")
        self.x_train = x_train
        self.y_train = y_train
        self.x_test = x_test
        self.y_test = y_test


    def Initialise_WandB_Sweep(self, FurtherTrain=False, ModelName = None):

        pprint.pprint(self.sweep_config)

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
            callback_config['EarlyStopping']["min delta"] = 1
            callback_config['EarlyStopping']["patience"] = 2000

        callback_config['SaveBest']["include"] = False

        callback_config['Val Loss Threshold']["include"] = True
        if callback_config['Val Loss Threshold']["include"]:
            callback_config['Val Loss Threshold']["Threshold"] = 0.05

        callback_config['Reduce lr']["include"] = True
        if callback_config['Reduce lr']["include"]:
            callback_config['Reduce lr']["factor"] = 0.5
            callback_config['Reduce lr']["patience"] = 500
            callback_config['Reduce lr']["min delta"] = 0.1
            callback_config['Reduce lr']["min lr"] = 0.00001


        callback_config['Print_nEpochs']["include"] = True
        if callback_config['Print_nEpochs']["include"]:
            callback_config['Print_nEpochs']["fequency"] = 100


        callback_config['wandb']["include"] = True
        self.callback_config = callback_config
                

        #------------------------------------------------------------------------
        def train_sweep(config=None):
            with wandb.init(config=config, project="SweepDir"):
            # If called by wandb.agent, as below,
            # this config will be set by Sweep Controller
                tf.keras.backend.clear_session()
                loss_func = wandb.config['loss_func']   
                if FurtherTrain:
                    ic("testing")
                    Model = load_model(r'C:\Users\vdwti\OneDrive - Imperial College London\YEAR 4\FYP\Scripts\Personal\FYP\DNN-Generation-Training\saved_models/'+ModelName)
                    NewModel = TrainOld(Model, self.x_train, self.y_train, self.x_test, self.y_test, config=wandb.config)

                elif FurtherTrain==False:
                    NewModel = TrainNew(self.x_train, self.y_train, self.x_test, self.y_test, config=wandb.config)

                    NewModel.Build(Inputs=tf.keras.Input(self.x_train.shape[1]), Dropout=[0,0])
                
                NewModel.Compile(loss_func = "mape")
                NewModel.Train(callback_config=self.callback_config, sweeping=True)
                ic("still working")
                
               
        self.train_sweep = train_sweep

    #------------------------------------------------------------------------
    #------------------------------------------------------------------------

    def Run_WandB_Sweep(self, count):
        wandb.agent(self.sweep_id, self.train_sweep, count=count)
        #------------------------------------------------------------------------
            
#------------------------------------------------------------------------
#------------------------------------------------------------------------
#------------------------------------------------------------------------

class ComparePerformance:
    def __init__(self, m=50):
        self.x_val = loadtxt(r'./Lambert_solutions/Lambert_Solutions_x_UNSEEN', delimiter = ',')
        self.y_val = loadtxt(r'./Lambert_solutions/Lambert_Solutions_y_UNSEEN', delimiter = ',')
        self.m = m

    def Compare(self):
        old_model = {} 

        for i, model in enumerate(os.scandir(r'.\saved_models')):
            print(model.path)
            old_model[i] = tf.keras.models.load_model(model.path)
            ic(model.name)
            old_model[i].evaluate(self.x_val, self.y_val)

        #def MaxPError(y_true, y_pred):
        #        return np.mean(np.abs((y_true-y_pred)/y_true), axis=1, keepdims=True)
        m = self.m
        n = range(m,m+5)
        y_true = self.y_val[n]
        ic(y_true)
        y_pred = old_model[i].predict(self.x_val[n])
        ic(y_pred)

#------------------------------------------------------------------------
#------------------------------------------------------------------------
#------------------------------------------------------------------------
