# FYP SUBMISSION SCRIPTS
This repository contains all code neccesary to replicate the results included in the master's thesis submitted in partial recognition of the MEng at Imperial College London

## List of requirements:
The list of python package requirements has been included in `requirements.txt`. All packages were installed on the Ubuntu OS - as such to avoid package installation issues please use WSL v2.0. This code has not been tested on Windows and its succesfull execution can not be guaranteed.

# Primary Scripts:

## `TFC_3D.py`
This code uses the theory of functional connections package to model the 3 dimensional Equations of Motion. `python3 TFC_3D.py -h` for positional arguments help

**Execution Command** : `python3 TFC_3D.py -n <NumRuNs> -J2 <IncludeJ2> -c <config>` 
  - <NumRuNs\>   [\-n]: Number of runs to be executed for each parameter configuration. Specify 1 if single run or any larger value in order to average out runtime or random initialisation
  - <IncludeJ2\> [\-J2]: Specifies whether to include J2 purturbation in calculations. Must tbe either 1 or 0
  - <config\>    [\-c]: Specify desired configuration to execute. Choose from:
      - _SingleTFC_ : A single lambert problem using TFC model
      - _SingleXTFC_: A single lambert problem using XTFC model
\     
      - _PolyOrder_ : Analyse the effect of Polynomial Order on TFC performance.      
      - _PolyRemove_: Analyse the effect of Polynomial Order Removed on TFC performance.
      - _PolyOrder_ : Analyse the effect of Number of Points on TFC performance.  
\      
      - _PolyOrderXTFC_ : Analyse the effect of Polynomial Order on XTFC performance.      
      - _PolyRemoveXTFC_: Analyse the effect of Polynomial Order Removed on XTFC performance.
      - _PolyOrderXTFC_ : Analyse the effect of Number of Points on XTFC performance.
\
      - _SeedsXTFC_ : Analyse the effect of RNG seed on XTFC performance.
      - _CompEffTFC_: Quantify computational efficiency of various TFC configurations.
      - _HeatMapTFC_: Quantify Loss performance for various TFC configurations.

## `DEEPXDE_PINN_training.py`
This code uses the DEEPXDE package to model either 1D, 2D or 3D equations of motion.

Note that the 2D and 3D versions converge sporadically for $\Delta t$/tnc \> 0.1

**Execution Command** : `DEEPXDE_PINN_training.py <RunType>` 
  - \<RunType\> : Specifies what problem to solve. Must be one of [ 1D, 2D , 3D ]

## `Visualise_Tools.py` and `OrbMech_Utilities.py`
A variety of tools and functions to aide in execution of the code. 
