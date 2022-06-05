# FYP SUBMISSION SCRIPTS
This repository contains all code neccesary to replicate the results included in the master's thesis submitted in partial recognition of the MEng at Imperial College London

## List of requirements:
The list of python package requirements has been included in `requirements.txt` and the anaconda environment export to `FYP.yml`.
All packages were installed on the Ubuntu OS - as such to avoid package installation issues please use WSL v2.0. This code has not been tested on Windows and its succesfull execution can not be guaranteed.

# Primary Scripts:

## `TFC_3D.py`
This code uses the theory of functional connections package to model the 3 dimensional Equations of Motion. 

```bash
python3 TFC_3D.py -h
```
for positional arguments help

**Execution Command** : 
```bash 
python3 TFC_3D.py -n <NumRuNs> -J2 <IncludeJ2> -c <config>
```` 
  - <NumRuNs\>   [\-n]: Number of runs to be executed for each parameter configuration. Specify 1 if single run or any larger value in order to average out runtime or random initialisation
  - <IncludeJ2\> [\-J2]: Specifies whether to include J2 purturbation in calculations. Must tbe either 1 or 0
  - <config\>    [\-c]: Specify desired configuration to execute. Choose from:
      - _SingleTFC_ : A single lambert problem using TFC model
      - _SingleXTFC_: A single lambert problem using XTFC model
     
      - _PolyOrder_ : Analyse the effect of Polynomial Order on TFC performance.      
      - _PolyRemove_: Analyse the effect of Polynomial Order Removed on TFC performance.
      - _PolyOrder_ : Analyse the effect of Number of Points on TFC performance.  
    
      - _PolyOrderXTFC_ : Analyse the effect of Polynomial Order on XTFC performance.      
      - _PolyRemoveXTFC_: Analyse the effect of Polynomial Order Removed on XTFC performance.
      - _PolyOrderXTFC_ : Analyse the effect of Number of Points on XTFC performance.

      - _SeedsXTFC_ : Analyse the effect of RNG seed on XTFC performance.
      - _CompEffTFC_: Quantify computational efficiency of various TFC configurations.
      - _HeatMapTFC_: Quantify Loss performance for various TFC configurations.

Additional configs may be added by following the template below.  Configs are defined on lines 600 onwards

```python 

if config=='SingleTFC':
    TrainingDf, TrainingStats = train_models(poly_orders=[50], points=[51], save_orbit=True, plot=True, run_type='TFC')
    
if config=='SingleXTFC':
    TrainingDf, TrainingStats = train_models(points=[200], poly_orders=[100], poly_removes=[-1], basis_funcs=['ELMTanh'], methods = ["lstsq"],  save_orbit=True, plot=True, run_type='XTFC')

```

## `DEEPXDE_PINN_training.py`
This code uses the DEEPXDE package to model either 1D, 2D or 3D equations of motion.

Note that the 2D and 3D versions converge sporadically for $\Delta t$/tnc \> 0.1

**Execution Command** : 
```bash 
python3 DEEPXDE_PINN_training.py <RunType>
```
  - \<RunType\> : Specifies what problem to solve. Must be one of [ 1D, 2D , 3D ]

The code below can be used to defined and alter the PINN model that is being trained.
```python
 net = dde.maps.FNN([1] + [50] * 3 + [1], "sigmoid", "Glorot uniform")
    net.apply_output_transform(
        lambda x, y: (ub-x)/ub*r0 + (x/ub)*rf + x/ub*(ub-x)/ub*y
    )
    model = dde.Model(data, net)
```

## `DNN_Training.py`
  This code used tensorflow to train a neural network using specified model configurations. It uses the WandB API to keep track of training runs and some alteration may be required to configure the code to run with the user's WandB account. Alternatively the sections of the code that require WandB can be commented out. 
  
 **Execution Command** : 
```bash 
python3 DNN_Training.py <RunType> [<NumRuns>]
```

  - \<RunType\> : Specifies whether to train specified model or conduct WandB hyperparameter sweep. Must be one of [_single_, _sweep_]
  - \<NumRuns\> : _Optional Parameter_. Specifies how many models to train if conducting WandB sweep. 

Configs must be declared on lines 83-104 in the following format:
```python
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
```
## `Visualise_Tools.py`,  `OrbMech_Utilities.py`, `DNN_Tools`
A variety of tools and functions to aide in execution of the code. 
