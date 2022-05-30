# FYP SUBMISSION SCRIPTS
This repository contains all code neccesary to replicate the results included in the master's thesis submitted in partial recognition of the MEng at Imperial College London

## List of requirements:
The list of python package requirements has been included in `requirements.txt`. All packages were installed on the Ubuntu OS - as such to avoid package installation issues please use WSL v2.0. This code has not been tested on Windows and its succesfull execution can not be guaranteed.

# Primary Scripts:

## `TFC_3D.py`
This code uses the theory of functional connections package to model the 3 dimensional Equations of Motion.

**Execution Command** : `python3 TFC_3D.py <NumRUNs> <IncludeJ2>` 
  - \<NumRUNs\> : Number of runs to be executed for each parameter configuration. Specify 1 if single run or any larger value in order to average out runtime or random     initialisation
  - \<IncludeJ2\> : Specifies whether to include J2 purturbation in calculations. Mus tbe either 1 or 0


## `DEEPXDE_PINN_training.py`
This code uses the DEEPXDE package to model either 1D, 2D or 3D equations of motion.

Note that the 2D and 3D versions converge sporadically for $\Delta t$/tnc \> 0.1

**Execution Command** : `DEEPXDE_PINN_training.py <RunType>` 
  - \<NumRUNs\> : Number of runs to be executed for each parameter configuration. Specify 1 if single run or any larger value in order to average out runtime or random     initialisation
  - \<RunType\> : Specifies what problem to solve. Must be [_'1D', '2D' , '3D_]
