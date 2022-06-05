import warnings
warnings.filterwarnings("ignore", message=r"Passing", category=FutureWarning)

import numpy as np
import pykep as pk
from numpy import loadtxt
from icecream import ic
import pandas as pd
import sklearn.model_selection as skl_model_selection   

def GenerateDataSet(Size, format, Normalise=False, ReturnDF=False, ReturnStatsDF=False): 
        
    Start_Epoch     = int(pk.epoch_from_string("2001-01-01 00:00:00").mjd)
    End_Epoch       = int(pk.epoch_from_string("2031-01-01 00:00:00").mjd)

    TOF_range       = [10,300]
    target_list     = ['mars', 'venus', 'mercury','jupiter']
    n_samples       = int(Size/len(target_list))        # number of Lambert Solutions desired

    lamsol_list     = []
    
    TOF             = np.empty((n_samples*len(target_list), 1))
    DeltaV          = np.empty((n_samples*len(target_list), 1))


    r0              = np.empty((n_samples*len(target_list), 3))
    rf              = np.empty((n_samples*len(target_list), 3))
    v0              = np.empty((n_samples*len(target_list), 3))
    vf              = np.empty((n_samples*len(target_list), 3))
    v1              = np.empty((n_samples*len(target_list), 3))
    v2              = np.empty((n_samples*len(target_list), 3))
    ooe0            = np.empty((n_samples*len(target_list), 6))
    ooef            = np.empty((n_samples*len(target_list), 6))
    mee0            = np.empty((n_samples*len(target_list), 6))
    meef            = np.empty((n_samples*len(target_list), 6))

    for (i,_target) in enumerate(target_list):
        ic(_target)

        t1              = np.random.randint(low=Start_Epoch, high=End_Epoch, size=(n_samples,1))
        _TOF             = np.random.randint(low=TOF_range[0], high=TOF_range[1], size=(n_samples,1))
        t2              = t1 + _TOF

        #Departure       = pk.planet(pk.epoch(54000,"mjd"),(9.99e-01 * pk.AU,1.67e-02, 8.85e-04 * pk.DEG2RAD, 1.75e+02 * pk.DEG2RAD, 2.87e+02 * pk.DEG2RAD, 2.57e+02 * pk.DEG2RAD), pk.MU_SUN, 398600e9, 6378000, 6900000,  'Earth')

        Departure       = pk.planet.jpl_lp('earth') 
        Target          = pk.planet.jpl_lp(_target)


        States0         = [Departure.eph(pk.epoch(int(time), 'mjd')) for time in t1]
        Statesf         = [Target.eph(pk.epoch(int(time), 'mjd')) for time in t2]

        _r0              = np.array([States[0] for States in States0])
        _rf              = np.array([States[0] for States in Statesf])
    
        _v0              = np.array([States[1] for States in States0])
        _vf              = np.array([States[1] for States in Statesf])

        #print(r0[i*n_samples : (i+1)*n_samples, 0:3].shape)

        r0[i*n_samples : (i+1)*n_samples, 0:3] = _r0
        rf[i*n_samples : (i+1)*n_samples, 0:3] = _rf

        v0[i*n_samples : (i+1)*n_samples, 0:3] = _v0
        vf[i*n_samples : (i+1)*n_samples, 0:3] = _vf

        for k in range(0,n_samples):

            ooe0[k+i*n_samples,:] = pk.ic2par(_r0[k,:], _v0[k,:], pk.MU_SUN)
            ooef[k+i*n_samples,:] = pk.ic2par(_rf[k,:], _vf[k,:], pk.MU_SUN)

            mee0[k+i*n_samples,:] = pk.ic2eq(_r0[k,:], _v0[k,:], pk.MU_SUN)
            meef[k+i*n_samples,:] = pk.ic2eq(_rf[k,:], _vf[k,:], pk.MU_SUN)

            lamsol_list.append(pk.lambert_problem(r1=_r0[k], r2=_rf[k], tof=int(_TOF[k])*24*3600,
                                            mu=pk.MU_SUN))
            v1[k+i*n_samples,:] = lamsol_list[k+i*n_samples].get_v1()[0]
            v2[k+i*n_samples,:] = lamsol_list[k+i*n_samples].get_v2()[0]


            DeltaV[k+i*n_samples,0] = (np.linalg.norm(np.subtract(v1,_v0[k])) + 
                        np.linalg.norm(np.subtract(v2,_vf[k])))
                            

                        
            TOF[k+i*n_samples,0] = _TOF[k]

    if ReturnDF:
        ''' 
        features to be included:
            rf - r0 : x,y,z
            ooef - ooe0 : all 6 elements
            meef - mee0 : all 6 elements
            v1 : x,y,z

        => 18 features

        ''' 
        columns = ['$\Delta R_{x}$', '$ \Delta R_{y}$', '$\Delta R_{z}$',
                             '$\Delta a$', '$\Delta e$','$\Delta i$','$\Delta \Omega$','$\Delta \omega$','$\Delta E$',
                             '$\Delta p$','$\Delta f$','$\Delta g$','$\Delta h$','$\Delta k$','$\Delta L$',
                             '$\Delta T$',
                             '$\Delta V_{1x}$', '$\Delta V_{1y}$', '$\Delta V_{1z}$']
        return pd.DataFrame(np.hstack([rf - r0, ooef - ooe0, meef - mee0, TOF, v1]), columns=columns)

    elif ReturnStatsDF: 
        columns = ['$R_{1x}$', '$ R_{1y}$', '$R_{1z}$',
                    '$R_{2x}$', '$ R_{2y}$', '$R_{2z}$',
                    'a0', 'e', 'i', '$\Omega', '$\Omega', 'E',
                    'af', 'e', 'i', '$\Omega', '$\Omega', 'E',
                    '$ \Delta T$',
                    '$ V_{1x}$', '$ V_{1y}$', '$ V_{1z}$']
        return pd.DataFrame(np.hstack([r0, rf, ooe0, ooef, TOF, v1]), columns=columns)
                
    else:
        if format=='kep':
            NN_Input = np.hstack((r0,rf,TOF))
        elif format=='ooe':
            NN_Input = np.hstack((ooe0, ooef, TOF))
        elif format=='mee':
            NN_Input = np.hstack((mee0, meef, TOF))

        NN_Output = v1
        ic(NN_Output.shape)


        x_train, x_test, y_train, y_test = skl_model_selection.train_test_split(NN_Input, NN_Output, test_size=0.2)
        ic(Normalise)

        if Normalise:
            ic("normalised")
            nc = np.linalg.norm(x_train[:,0:3], axis=1, keepdims=True)
            tnc = np.sqrt(nc**3/pk.MU_SUN)

            x_train[:,0:3] /= nc
            x_train[:,3:6] /= nc
            x_train[:,6:7] /= tnc

            nc_test = np.linalg.norm(x_test[:,0:3], axis=1, keepdims=True)
            tnc_test = np.sqrt(nc_test**3/pk.MU_SUN)

            x_test[:,0:3] /= nc_test
            x_test[:,3:6] /= nc_test
            x_test[:,6:7] /= tnc_test

            ic(y_train[:,0], nc, tnc)
            y_train /= nc/tnc
            y_test /= nc_test/tnc_test
        
        return x_train, y_train, x_test, y_test
    
def ReturnDataSet(OldData=True, DataSetSize=20000, format='kep', Normalise=False, ReturnDF=False, ReturnStatsDF=False):

    if OldData:  
        random_indices = np.random.choice(100000, size=DataSetSize, replace=False)
        NN_Input = loadtxt(r'./Lambert_Solutions/Lambert_Solutions_x', delimiter = ',')[random_indices,:]
        NN_Output = loadtxt(r'./Lambert_Solutions/Lambert_Solutions_y', delimiter = ',')[random_indices,:]

        x_train, x_test, y_train, y_test = skl_model_selection.train_test_split(NN_Input, NN_Output, test_size=0.2)

    else:
        if ReturnDF:
            ic("returning DF")
            return GenerateDataSet(DataSetSize, format=format, Normalise=Normalise, ReturnDF=ReturnDF)
        elif ReturnStatsDF:
            ic("returning Statistics DF")
            return GenerateDataSet(DataSetSize, format=format, Normalise=Normalise, ReturnStatsDF=ReturnStatsDF)
        else:
            x_train, y_train, x_test, y_test= GenerateDataSet(DataSetSize, format=format, Normalise=Normalise, ReturnDF=ReturnDF)

    ic(x_train.shape)

    m_train = x_train.shape[0]
    num_states = x_train.shape[1]
    m_test = x_test.shape[0]

    ic (m_train)
    ic (m_test)
    print ("Each input is of size: (1, " + str(x_train.shape[1]) + ")")
    ic (x_train.shape)
    print ("Each Output is of size: (1, " + str(y_train.shape[1]) + ")")
    ic (y_train.shape)


    return x_train, y_train, x_test, y_test