from ctypes.wintypes import BOOLEAN
import sys

import jax.numpy         as np
import numpy             as onp
import matplotlib.pyplot as plt
import pykep             as pk
import time
import pickle
import pandas            as pd
import time

from tfc              import utfc
from tfc.utils        import TFCDict, egrad, MakePlot, NllsClass, NLLS, LS
from jax              import jit, jacfwd
from jax.numpy.linalg import norm

from icecream         import ic
from scipy.integrate  import odeint
from IPython.display  import display 
from Visualise_Tools  import set_size, format_axes
from OrbMech_Utilities import plot3D_grav, Get_PropagationError
from matplotlib.ticker import PercentFormatter
from os.path import exists

from astropy import units as u

from poliastro.bodies              import Earth, Mars, Sun

import warnings
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)


# print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))

# Arguments passed
n = len(sys.argv) 
print("\nName of Python script:", sys.argv[0])

import argparse

parser=argparse.ArgumentParser(
    description='''My Description. And what a lovely description it is. ''',
    epilog="""All is well that ends well.""")

parser.add_argument('-n','--num_runs', type=int, default=1, help='Number of Iterations per training config')
parser.add_argument('-J2','--Include_J2', type=int, default=0, choices={0,1}, help='Specify whether to include J2 perturbations in analysis!')
parser.add_argument('-c','--config', type=str, default={'SingleTFC'}, choices={'SingleTFC', 
                                                                               'SingleXTFC', 
                                                                               'SeedsXTFC', 
                                                                               'PolyOrderTFC',
                                                                               'CompEffTFC',
                                                                               'HeatMapTFC',
                                                                               'PolyRemoveTFC',
                                                                               'PointsTFC',
                                                                               'PolyOrderXTFC',
                                                                               'PolyRemoveXTFC',
                                                                               'PointsXTFC'}, help='Specify what program program you wish to run')
args=parser.parse_args()

num_runs = args.num_runs
Include_J2 = args.Include_J2
config = args.config

# num_runs = int(sys.argv[2]) # number of runs for each config
# Include_J2 = float(sys.argv[3]) # include J2 perturbation?

class LambertEq():
    def __init__(self, print=True):
        # Reset workspace 
        self.print=print
        onp.random.seed()
        # tf.compat.v1.reset_default_graph()
    class color:
        PURPLE = '\033[95m'
        CYAN = '\033[96m'
        DARKCYAN = '\033[36m'
        BLUE = '\033[94m'
        GREEN = '\033[92m'
        YELLOW = '\033[93m'
        RED = '\033[91m'
        BOLD = '\033[1m'
        UNDERLINE = '\033[4m'
        RESET = '\033[0m'


class LambertEq(LambertEq):
    def Get_Lambert_Sun(self, new=True, shortway=True, defined=False):
        
        ############################################################################
        ########################## Define Lambert Problem ##########################
        ############################################################################

        if new and defined:
            # Set limits on TOF and starting date
            Start_Epoch     = int(pk.epoch_from_string("2020-01-26 00:00:00").mjd)
            End_Epoch       = int(pk.epoch_from_string("2020-05-18 00:00:00").mjd)
            
            # specify target planet
            _target = 'mars'
            
            self.t1 = onp.array([Start_Epoch])
            self.t2 = End_Epoch
            self._TOF = self.t2-self.t1

        else:

            # Set limits on TOF and starting date
            Start_Epoch     = int(pk.epoch_from_string("2001-01-01 00:00:00").mjd)
            End_Epoch       = int(pk.epoch_from_string("2031-01-01 00:00:00").mjd)
            TOF_range       = [20, 300]
            
            # specify target planet
            _target = 'mars'

            # only generate new problem if required
            if new:
                self.t1              = onp.random.randint(low=Start_Epoch, high=End_Epoch, size=1)
                self._TOF            = onp.random.randint(low=TOF_range[0], high=TOF_range[1], size=1)
                self.t2              = self.t1 + self._TOF

        # Get Ephemeris data from pykep
        Departure       = pk.planet.jpl_lp('earth') 
        Target          = pk.planet.jpl_lp(_target)

        States0         = Departure.eph(pk.epoch(int(self.t1), 'mjd'))
        Statesf         = Target.eph(pk.epoch(int(self.t2), 'mjd'))

        self._r0              = onp.array(States0[0])
        self._rf              = onp.array(Statesf[0])
        self._v0              = onp.array(States0[1])
        self._vf              = onp.array(Statesf[1])

        if shortway:
            self.clockwise = True if onp.cross(self._r0,self._rf)[2] < 0 else False
        else:
            self.clockwise = True if onp.cross(self._r0,self._rf)[2] >= 0 else False


        ################################################################################
        ############################# Solve Lambert Problem ############################
        ################################################################################

        lamsol_list = pk.lambert_problem(r1=self._r0, r2=self._rf, tof=int(self._TOF)*24*3600,
                                        mu=pk.MU_SUN, cw=self.clockwise)
        self.v1 = lamsol_list.get_v1()[0]
                            
        self.TOF = self.t2-self.t1

        ################################################################################
        ############################# Set up training data #############################
        ################################################################################


        # Define normamlising constants for distance and time
        self.nc = float(onp.linalg.norm(self._r0))
        self.tnc = float(onp.sqrt(self.nc**3/pk.MU_SUN))

        # Normalise input data
        self.r0 = (self._r0/self.nc).astype(float)
        self.rf = (self._rf/self.nc).astype(float)
        self.mu = float(pk.MU_SUN/(self.nc)**3 * self.tnc**2)

        # Specifiy short or long way solution (dtheta<180?)
        self.short_way=shortway

        ################################################################################
        ############################# Ensure solution exists ###########################
        ################################################################################


        c = norm(self._r0 - self._rf)
        s = (norm(self._r0) +  norm(self._rf) + c) /2

        alpha_m = onp.pi
        beta_m = 2*onp.arcsin(onp.sqrt((s-c)/s))

        # Minimum Energy Solution - determines long or short time solution 
        dt_m = np.sqrt(s**3/(8*pk.MU_SUN))*(onp.pi - beta_m + onp.sin(beta_m))     
        dtheta = onp.arccos(onp.dot(self._r0,self._rf)/(norm(self._r0)*norm(self._rf)))

        # if long way specified, adjust change in true anomaly for parabolic transfer time calculation
        if not self.short_way:
            print('Adjusting for long way solution')
            dtheta = 2*onp.pi - dtheta

        # parabolic transfer time - minimum 
        dt_p = onp.sqrt(2)/3*(s**1.5 - onp.sign(onp.sin(dtheta))*(s-c)**1.5)/onp.sqrt(pk.MU_SUN)
    
        ############################################################################
        ############################# Get Correct SMA ##############################
        ############################################################################

        if new and self.print:
            print(self.color.BOLD + self.color.GREEN + 
                f'Start Date: {pk.epoch(int(self.t1), "mjd")}')
            print(f'End Date:   {pk.epoch(int(self.t2), "mjd")}\n' + self.color.RESET)

            print(f'Min E deltaT: {dt_m/3600/24:.3f} days')
            print(f'parabolic deltaT: {dt_p/3600/24:.3f} days')
            print(f'Desired TOF:  {self.TOF[0]} days\n')


def TrainModel(l, ub, points=100, poly_order=30, poly_removed=2, basis_func='CP', method="pinv", plot=True, save_orbit=False, run_type='TFC'):

    ########### Define Constants ################
    mu = l.mu
    J2 = Sun.J2.value
    
    R_sun = (6963408*1e3) /l.nc # sun radius in m

    r0 = l.r0
    rf = l.rf
    v0 = onp.array(l._v0) # original velocity
    v1 = l.v1 # lambert velocity

    _deltat = l.TOF*24*3600
    ub = float(_deltat/l.tnc)

    # start = time.time()
    # Create the univariate TFC class
    N = points # Number of points in the domain
    m = poly_order # Degree of basis function expansion
    nC = poly_removed # Indicates which basis functions need to be removed from the expansion

    start = time.time_ns()/(10 ** 9)

    myTfc = utfc(N,nC,m, basis=basis_func, x0=0,xf=ub)
    x = myTfc.x # Collocation points from the TFC class

    # Get the basis functions from the TFC class
    H = myTfc.H
    H0 = lambda x: H(np.zeros_like(x))
    Hf = lambda x: H(ub*np.ones_like(x))

    #------------------------------------------------------------------------
    #------------------------------------------------------------------------
    #--------------------- Create the constrained expression ----------------
    #------------------------------------------------------------------------
    #------------------------------------------------------------------------

    g = lambda x,xi: np.array([np.dot(H(x),xi[:,0])])

    rx = lambda x,xi: np.dot(H(x),xi['x'])  \
                + (ub-x)/ub*( r0[0] - np.dot(H0(x),xi['x']) ) \
                + (x)/ub*( rf[0] - np.dot(Hf(x),xi['x']) ) \

    ry = lambda x,xi: np.dot(H(x),xi['y'])  \
                + (ub-x)/ub*( r0[1] - np.dot(H0(x),xi['y']) ) \
                + (x)/ub*( rf[1] - np.dot(Hf(x),xi['y']) ) \

    rz = lambda x,xi: np.dot(H(x),xi['z'])  \
                + (ub-x)/ub*( r0[2] - np.dot(H0(x),xi['z']) ) \
                + (x)/ub*( rf[2] - np.dot(Hf(x),xi['z']) ) \

    # Create the residual
    drx = egrad(rx)
    d2rx = egrad(drx)

    dry = egrad(ry)
    d2ry = egrad(dry)

    drz = egrad(rz)
    d2rz = egrad(drz)
    t0 = onp.zeros_like(x)

    v1_guess = lambda x, xi: np.array([drx(t0, xi), dry(t0, xi), drz(t0, xi)])[np.array([0,1,2]), np.array([0,0,0])]*l.nc/l.tnc
    v_angle  = lambda x, xi: np.arccos(np.dot(v1_guess(x,xi),v0)/(norm(v0)*norm(v1_guess(x,xi))))*180/np.pi

    #------------------------------------------------------------------------
    #------------------------------------------------------------------------
    #---------------------------- Create ODE --------------------------------
    #------------------------------------------------------------------------
    #------------------------------------------------------------------------

    if 0:
        penalty = 2.0
        if  not l.clockwise:
            print('Option 1')
            res_rx = lambda x, xi: d2rx(x,xi) + mu*rx(x,xi)/((rx(x, xi)**2 +  ry(x,xi)**2 + rz(x,xi)**2)**(3/2)) + penalty*(v_angle(x,xi)>90)
            res_ry = lambda x, xi: d2ry(x,xi) + mu*ry(x,xi)/((rx(x, xi)**2 +  ry(x,xi)**2 + rz(x,xi)**2)**(3/2)) + penalty*(v_angle(x,xi)>90)
            res_rz = lambda x, xi: d2rz(x,xi) + mu*rz(x,xi)/((rx(x, xi)**2 +  ry(x,xi)**2 + rz(x,xi)**2)**(3/2)) + + penalty*(v_angle(x,xi)>90)

        else: 
            print('Option 2')
            res_rx = lambda x, xi: abs(d2rx(x,xi) + mu*rx(x,xi)/((rx(x, xi)**2 +  ry(x,xi)**2 + rz(x,xi)**2)**(3/2))) + penalty*(v_angle(x,xi)<90)
            res_ry = lambda x, xi: abs(d2ry(x,xi) + mu*ry(x,xi)/((rx(x, xi)**2 +  ry(x,xi)**2 + rz(x,xi)**2)**(3/2))) + penalty*(v_angle(x,xi)<90)
            res_rz = lambda x, xi: abs(d2rz(x,xi) + mu*rz(x,xi)/((rx(x, xi)**2 +  ry(x,xi)**2 + rz(x,xi)**2)**(3/2))) + penalty*(v_angle(x,xi)<90)

    else:
        r_norm = lambda x, xi: (rx(x, xi)**2 +  ry(x,xi)**2 + rz(x,xi)**2)**(1/2)
        res_rx = lambda x, xi: d2rx(x,xi) + mu*rx(x,xi)/(r_norm(x,xi)**3) * ( 1.0 + Include_J2*1.5*J2*(R_sun/(r_norm(x,xi))**2)*(1-5*(rz(x,xi)/r_norm(x,xi))**2) )
        res_ry = lambda x, xi: d2ry(x,xi) + mu*ry(x,xi)/(r_norm(x,xi)**3) * ( 1.0 + Include_J2*1.5*J2*(R_sun/(r_norm(x,xi))**2)*(1-5*(rz(x,xi)/r_norm(x,xi))**2) )
        res_rz = lambda x, xi: d2rz(x,xi) + mu*rz(x,xi)/(r_norm(x,xi)**3) * ( 1.0 + Include_J2*1.5*J2*(R_sun/(r_norm(x,xi))**2)*(3-5*(rz(x,xi)/r_norm(x,xi))**2) )

    L = jit(lambda xi: np.hstack([res_rx(x, xi), res_ry(x,xi), res_rz(x,xi)]))

    #------------------------------------------------------------------------
    #------------------------------------------------------------------------
    #------------------------ Minimize the residual -------------------------
    #------------------------------------------------------------------------
    #------------------------------------------------------------------------

    # set up weights dict
    xi = TFCDict({'x':onp.zeros(H(x).shape[1]), 'y':onp.zeros(H(x).shape[1]), 'z':onp.zeros(H(x).shape[1])})
    
    # set up initial guess
    # xi['x'] = onp.dot(onp.linalg.pinv(jacfwd(rx,1)(np.array([0]),xi)['x']),r0[0:1]-rx(np.array([0]),xi))
    # xi['y'] = onp.dot(onp.linalg.pinv(jacfwd(ry,1)(np.array([0]),xi)['y']),r0[1:2]-ry(np.array([0]),xi))
    # xi['z'] = onp.dot(onp.linalg.pinv(jacfwd(rz,1)(np.array([0]),xi)['z']),r0[2:3]-rz(np.array([0]),xi))

    # Create NLLS class
    # nlls = NllsClass(xi,L,timer=True)
    # xi,_,time = nlls.run(xi)

    xi,_,Time = NLLS(xi,L, maxIter=200, method=method, timer=True)
    runtime = time.time_ns()/(10 ** 9) - start

    print(f'CPU Run time = {Time}')
    print(f'Real Run time = {runtime}')


    # ic(v1_guess(t0,xi))
    # ic(v1, l.clockwise, v_angle(t0,xi), norm(L(xi)))    


    # run_time = time.time() - start

    #------------------------------------------------------------------------
    #------------------------------------------------------------------------
    #------------------- Calculate the error on the test set ----------------
    #------------------------------------------------------------------------
    #------------------------------------------------------------------------

    time_Extend = 1.05

    testSet = np.linspace(0,time_Extend*ub,100)
    # error = np.abs(ry(testSet,xi)-realSoln(testSet))
    predicted_velocity_x = (drx(np.array([0.0]),xi)*l.nc/l.tnc)[0]
    vel_error_x = (v1[0] - predicted_velocity_x) / v1[0]

    predicted_velocity_y = (dry(np.array([0.0]),xi)*l.nc/l.tnc)[0]
    vel_error_y = (v1[1] - predicted_velocity_y) / v1[1]

    predicted_velocity_z = (drz(np.array([0.0]),xi)*l.nc/l.tnc)[0]
    vel_error_z = (v1[2] - predicted_velocity_z) / v1[2]

    print(f'initial velocity error = {vel_error_x}, {vel_error_y}, {vel_error_z}%')

    ### Calculate Residual Error ###
    Error_testSet = np.linspace(0,time_Extend*ub,1000)
    residuals_x = res_rx(Error_testSet, xi)
    residuals_y = res_ry(Error_testSet, xi)
    residuals_z = res_rz(Error_testSet, xi)
    

    #------------------------------------------------------------------------
    #------------------------------------------------------------------------
    #--------------------- Calculate Position Error  ------------------------
    #------------------------------------------------------------------------
    #------------------------------------------------------------------------
    if save_orbit:

        v1_pred = np.array([predicted_velocity_x, predicted_velocity_y, predicted_velocity_z])
        with open('OrbitParams.npy','wb') as file:
            np.save(file, l._r0)
            np.save(file, l._rf)
            np.save(file, l._v0)
            np.save(file, l._vf)

            np.save(file, l.v1)
            np.save(file, v1_pred)
            
            np.save(file, l.TOF)
            np.save(file, l.t1)

        pk_error, TFC_error, error_ratio = Get_PropagationError()

    states  = plot3D_grav(r0*l.nc, v1,              ub*l.tnc, uf=rf*l.nc, J2 = J2, R = R_sun*l.nc)
    states2 = plot3D_grav(r0*l.nc, v1_guess(t0,xi), ub*l.tnc, uf=rf*l.nc, J2 = J2, R = R_sun*l.nc)

    predict_rf = states[-1,np.array([0,2,4])]
    predict_rf2 = states2[-1,np.array([0,2,4])]

    # print('\n')
    # print(f'rf target     : {rf*l.nc}')
    # print(f'rf with PyKEP : { states[-1,np.array([0,2,4])]}')
    # print(f'rf with TFC   : { predict_rf2}')



    print(f'PyKEP final position error: {norm(predict_rf - rf*l.nc)/1000} km')
    print(f'Model final position error: {norm(predict_rf2 - rf*l.nc)/1000} km')

    #------------------------------------------------------------------------
    #------------------------------------------------------------------------
    #------------------------------ Graphing --------------------------------
    #------------------------------------------------------------------------
    #------------------------------------------------------------------------ 
    
    if plot:
        plt.style.use('tex')
        plt.rcParams['figure.dpi'] = 300
        plt.rcParams['savefig.dpi'] = 300

        ### Set Plot Name ###
        if Include_J2:
            if run_type=='TFC':
                posplot_image_name = 'J2_PositionPlot'
                poserror_image_name = 'J2_PositionError'
                residual_image_name = 'J2_residuals'
            elif run_type=='XTFC':
                posplot_image_name = 'J2_PositionPloXTFCt'
                poserror_image_name = 'J2_PositionErrorXTFC'
                residual_image_name = 'J2_residualsXTFC'
        else:
            if run_type=='TFC':
                posplot_image_name = 'PositionPlot'
                poserror_image_name = 'PositionError'
                residual_image_name = 'residuals'
            elif run_type=='XTFC':
                posplot_image_name = 'PositionPloXTFC'
                poserror_image_name = 'PositionErrorXTFC'
                residual_image_name = 'residualsXTFC'
        
        ### Position Plot ###
        fig = plt.figure(figsize=set_size(483.69687*1.05, 1, subplots = (1,1)))
        ax = plt.gca()

        states, t = plot3D_grav(r0*l.nc, v1, time_Extend*ub*l.tnc, uf=rf*l.nc, ax=ax, J2 = J2, R = R_sun*l.nc, return_t_array=True, Include_J2=Include_J2)
        ax.plot(rx(testSet,xi)*l.nc/1000,ry(testSet,xi)*l.nc/1000, 'ro', markersize=3 ,label='TFC Solution')
        format_axes(ax=ax, fontsize=12, xlabel = r'$R_{x}$ [km]', ylabel=r'$R_{y}$ [km]')
        fig.savefig('./Plot/'+posplot_image_name+'.pdf', bbox_inches='tight')
        

        ### Position Error ###
        pos_x = rx(t/l.tnc, xi)*l.nc
        pos_y = ry(t/l.tnc, xi)*l.nc
        pos_z = rz(t/l.tnc, xi)*l.nc
        x_error = (pos_x - states[:,0])/states[:,0]
        y_error = (pos_y - states[:,2])/states[:,2]
        z_error = (pos_z - states[:,4])/states[:,4]

        fig2 = plt.figure(figsize=set_size(483.69687*1.05, 1, subplots = (2,1)))
        ax = plt.gca()
        ax.semilogy(t/l.tnc/ub, onp.abs(x_error), 'b+', label=r'$R_{x}$ error', markersize=3, markeredgewidth=0.01)
        ax.semilogy(t/l.tnc/ub, onp.abs(y_error), 'g+', label=r'$R_{y}$ error', markersize=3, markeredgewidth=0.01)
        ax.semilogy(t/l.tnc/ub, onp.abs(z_error), 'r+', label=r'$R_{z}$ error', markersize=3, markeredgewidth=0.01)
        format_axes(ax=ax, fontsize=20, xlabel = 'time [ND]', ylabel=r'Relative Error Magnitude', scale_legend=True)
        ax.grid(which='major', color='#DDDDDD', linewidth=0.8)
        fig2.savefig('./Plot/'+poserror_image_name+'.pdf', bbox_inches='tight')


        ### Residual Error ###

        fig3 = plt.figure(figsize=set_size(483.69687*1.05, 1, subplots = (2,1)))
        ax = plt.gca()
        ax.semilogy(Error_testSet/ub, (onp.abs(residuals_x)), 'b+', label='x residual', markersize=3, markeredgewidth=0.01)
        ax.semilogy(Error_testSet/ub, (onp.abs(residuals_x)), 'bx', markersize=3, markeredgewidth=0.01)

        ax.semilogy(Error_testSet/ub, (onp.abs(residuals_y)), 'g+', label='y residual', markersize=3,  markeredgewidth=0.01)
        ax.semilogy(Error_testSet/ub, (onp.abs(residuals_y)), 'gx', markersize=3, markeredgewidth=0.01)

        ax.semilogy(Error_testSet/ub, (onp.abs(residuals_z)), 'r+', label='z residual', markersize=3,  markeredgewidth=0.1)
        ax.semilogy(Error_testSet/ub, (onp.abs(residuals_z)), 'rx', markersize=3,  markeredgewidth=0.1)
        # Show the major grid and style it slightly.
        ax.grid(which='major', color='#DDDDDD', linewidth=0.8)
        format_axes(ax=ax, fontsize=20, xlabel = 'time [ND]', ylabel=r'Absolute Error Magnitude', scale_legend=True)

        fig3.savefig('./Plot/'+residual_image_name+'.pdf', bbox_inches='tight')

    if Include_J2:
        CompEff = TFC_error*Time
        return [pk_error, TFC_error, error_ratio], runtime, CompEff
    else:
        CompEff = float(norm([vel_error_x, vel_error_y, vel_error_z]))*Time
        return float(norm([vel_error_x, vel_error_y, vel_error_z])), runtime, CompEff



def train_models(points=[51], poly_orders=[50], poly_removes=[2], basis_funcs=['LeP'], methods = ["pinv"], plot=False, save_orbit=False, run_type='TFC'):
    data = []
    total_start = time.time_ns()/(10 ** 9)


    for way in [True]:
        for poly_order in poly_orders:
            for poly_remove in poly_removes:
                for basis_func in basis_funcs:
                    for method in methods:
                        for point in points:
                            for i in range(num_runs): 
                                print(f'\nTraining {way, poly_order, poly_remove, basis_func, method, point, i}')
                                l = LambertEq(print=False)
                                l.Get_Lambert_Sun(shortway=way, defined=True)

                                _deltat = l.TOF*24*3600
                                ub = float(_deltat/l.tnc)

                                error, runtime, CompEff = TrainModel(l=l, ub=ub, 
                                                            points=point, 
                                                            poly_order=poly_order, 
                                                            poly_removed=poly_remove, 
                                                            basis_func=basis_func, 
                                                            method = method, 
                                                            plot=plot, 
                                                            save_orbit=save_orbit,
                                                            run_type=run_type)


                                if Include_J2:
                                    data.append([way, poly_order, point, poly_remove, basis_func,  method, i, error[0], error[1], error[2], runtime, CompEff])
                                else:
                                    data.append([way, poly_order, point, poly_remove, basis_func,  method, i, error, runtime, CompEff])
                                plt.close('all')


    total_runtime = time.time_ns()/(10 ** 9) - total_start
    print(f'Traing took {total_runtime} seconds' )

    if Include_J2:
        TrainingDf = pd.DataFrame(data, columns=['Shortway', 'poly_order','points', 'poly_remove', 'basis_function', 'method',
                                                 'Example', 'PK Loss', 'TFC Loss', 'Loss Ratio','Training Time', 'CompEff'])

        print('\n')
        display(TrainingDf)
        print('\n')

        TrainingStats = TrainingDf.groupby(['Shortway', 'basis_function', 'method', 'poly_order', 'poly_remove'])[['PK Loss', 'TFC Loss', 'Loss Ratio','Training Time','CompEff']].median()

        TrainingStats['PK Loss'] = TrainingStats['PK Loss'].map(lambda x: '%.5e' % x)
        TrainingStats['TFC Loss'] = TrainingStats['TFC Loss'].map(lambda x: '%.5e' % x)
        TrainingStats['Loss Ratio'] = TrainingStats['Loss Ratio'].map(lambda x: '%.5e' % x)

        print('\n')
        display(TrainingStats)
        # TrainingStats.to_pickle("Training_DF")
        print('\n')

    else:

        TrainingDf = pd.DataFrame(data, columns=['Shortway', 'poly_order', 'points', 'poly_remove', 'basis_function', 'method',
                                                'Example', 'Loss', 'Training Time', 'CompEff'])

        TrainingDf['Passed 1e-10'] = TrainingDf['Loss'].map(lambda x: 1.0*(x<1e-10))
        TrainingDf['Passed 1e-12'] = TrainingDf['Loss'].map(lambda x: 1.0*(x<1e-12))
        TrainingDf['Passed 1e-13'] = TrainingDf['Loss'].map(lambda x: 1.0*(x<1e-13))

        print('\n')
        display(TrainingDf)
        print('\n')

        TrainingStats = TrainingDf.groupby(['Shortway', 'basis_function', 'method', 'poly_order', 'poly_remove'])['Loss', 'Training Time', 'CompEff'].median()

        TrainingStats[['Percent 1e-10', 
                    'Percent 1e-12', 
                    'Percent 1e-13']] = TrainingDf.groupby(['Shortway', \
                                                            'basis_function', \
                                                            'method', \
                                                            'poly_order', \
                                                            'poly_remove'])[['Passed 1e-10', 'Passed 1e-12', 'Passed 1e-13']].mean()*100.0
        # TrainingStats[['Percent 10', 'Percent 0.1', 'Percent 0.001']] = TrainingStats[['Percent 10', 'Percent 0.1', 'Percent 0.001']].apply(lambda x: '%.2f' % x)
        TrainingStats['Loss'] = TrainingStats['Loss'].map(lambda x: '%.5e' % x)

        print('\n')
        display(TrainingStats)
        # TrainingStats.to_pickle("Training_DF")
        print('\n')

    return TrainingDf, TrainingStats

#------------------------------------------------------------------------
#------------------------------------------------------------------------
#--------------------- Execute single TFC or XTFC run -------------------
#------------------------------------------------------------------------
#------------------------------------------------------------------------


if config=='SingleTFC':
    TrainingDf, TrainingStats = train_models(poly_orders=[50], points=[51], save_orbit=True, plot=True, run_type='TFC')
if config=='SingleXTFC':
    TrainingDf, TrainingStats = train_models(points=[100], poly_orders=[100], poly_removes=[-1], basis_funcs=['ELMTanh'], methods = ["lstsq"],  save_orbit=True, plot=True, run_type='XTFC')


# In[1]:
#------------------------------------------------------------------------
#------------------------------------------------------------------------
#---------------------- Sweep over configurations -----------------------
#------------------------------------------------------------------------
#------------------------------------------------------------------------
# 
### Sweep over random seeds ### 
if config=='SeedsXTFC':

    TrainingDf, TrainingStats = train_models(points=[200], poly_orders=[100], poly_removes=[-1], basis_funcs=['ELMTanh'], methods = ["lstsq"],  save_orbit=False, plot=False)
    bins = onp.logspace(-11, -4, 14)
    ticks = 10**(onp.linspace(-11, -4, 8))

    if exists('Seed_Training_DF.pkl'):
        with open('Seed_Training_DF.pkl', 'rb') as file:
            df = pickle.load(file)
            TrainingDf = pd.concat([df, TrainingDf])
            TrainingDf.to_pickle('Seed_Training_DF.pkl')

    else:
        TrainingDf.to_pickle('Seed_Training_DF.pkl')
    
        # TrainingStats.to_pickle("Seed_Training_DF")
    weights=onp.ones((len(TrainingDf['Loss']),1)) / len(TrainingDf['Loss'])
    

    fig = plt.figure(figsize=set_size(483.69687*1.05, 1, subplots = (1,1)))
    ax = plt.gca()
    ax.hist(TrainingDf['Loss'], bins, weights=weights)
    ax.set_xscale('log')
    ax.yaxis.set_major_formatter(PercentFormatter(1))
    format_axes(ax=ax, fontsize=12, xlabel = r'Relative Error Magnitude', ylabel=r'Percent Count', force_ticks=ticks)

    ax2 = ax.twinx()
    ax2.hist(TrainingDf['Loss'], bins, histtype='step', cumulative=True, weights=weights, color='orange')
    ax2.yaxis.set_major_formatter(PercentFormatter(1))
    ax2.tick_params(axis='y', labelcolor='orange')
    format_axes(ax=ax2, fontsize=12, xlabel = r'Relative Error Magnitude', ylabel=r'Cumulative Percent Count', force_ticks=ticks)
    
    fig.savefig('./Plot/Seed_Error.pdf', bbox_inches='tight')

### Sweep over polynomial orders ###
if config=='PolyOrderTFC':
    poly_orders=list(range(2,200, 2))
    TrainingDf, TrainingStats = train_models(poly_orders=poly_orders, points=[200])

    fig = plt.figure(figsize=set_size(483.69687*1.05, 1, subplots = (2,1)))
    ax = plt.gca()
    ax.semilogy(poly_orders, TrainingDf['Loss'], 'b+', markersize=5, markeredgewidth=0.01)
    ax.grid(which='major', color='#DDDDDD', linewidth=0.8)
    format_axes(ax=ax, fontsize=20, xlabel = 'Polynomial Order', ylabel=r'Relative Error Magnitude', scale_legend=True)
    fig.savefig('./Plot/poly_orders.pdf', bbox_inches='tight')

### Find Computational Efficiency ###
if config=='CompEff':
    poly_orders=list(range(2,200, 10))
    TrainingDf, TrainingStats = train_models(poly_orders=poly_orders, points=[200])

    fig = plt.figure(figsize=set_size(483.69687*1.05, 1, subplots = (2,1)))
    ax = plt.gca()
    ax.semilogy(poly_orders, TrainingDf['CompEff'], 'b+', markersize=5, markeredgewidth=0.01)
    ax.grid(which='major', color='#DDDDDD', linewidth=0.8)
    format_axes(ax=ax, fontsize=20, xlabel = 'Polynomial Order', ylabel=r'Relative Error Magnitude', scale_legend=True)
    fig.savefig('./Plot/CompEff.pdf', bbox_inches='tight')

### Make Heat Map for Loss ###
if config=='HeatmapTFC':
    poly_orders=list(range(2,200, 1))
    points=list(range(3,200, 1))

    TrainingDf, TrainingStats = train_models(poly_orders=poly_orders, points=points)
    pivot = pd.pivot_table(TrainingDf, values='Loss', index = 'poly_order', columns='points')

    if exists('PivotTable.pkl'):
        with open('PivotTable.pkl', 'rb') as file:
            df = pickle.load(file)
            pivot = pd.concat([df, pivot])
            pivot.to_pickle('PivotTable.pkl')

    else:
        pivot.to_pickle('PivotTable.pkl')

    display(pivot)


    fig = plt.figure(figsize=set_size(483.69687*1.05, 1, subplots = (2,1)))
    ax = plt.gca()
    ax.imshow(pivot)
    ax.grid(which='major', color='#DDDDDD', linewidth=0.8)
    format_axes(ax=ax, fontsize=20, xlabel = 'Polynomial Order', ylabel=r'Relative Error Magnitude', scale_legend=True)
    fig.savefig('./Plot/Heatmap.pdf', bbox_inches='tight')

### Make Heat Map for CompEff ###
if config=='HeatmapCompEff':
    poly_orders=list(range(2,200, 5))
    points=list(range(3,200, 5))

    TrainingDf, TrainingStats = train_models(poly_orders=poly_orders, points=points, methods=['lstsq'], basis_funcs=['ELMTanh'])
    TrainingDf.to_pickle("Point_Order_SweepELM.pkl")
    pivot = pd.pivot_table(TrainingDf, values='CompEff', index = 'poly_order', columns='points')

    if exists('PivotTable_CE.pkl'):
        with open('PivotTable_CE.pkl', 'rb') as file:
            df = pickle.load(file)
            pivot = pd.concat([df, pivot])
            pivot.to_pickle('PivotTable_CE.pkl')

    else:
        pivot.to_pickle('PivotTable_CE.pkl')

    display(pivot)


    fig = plt.figure(figsize=set_size(483.69687*1.05, 1, subplots = (2,1)))
    ax = plt.gca()
    ax.imshow(pivot)
    ax.grid(which='major', color='#DDDDDD', linewidth=0.8)
    format_axes(ax=ax, fontsize=20, xlabel = 'Polynomial Order', ylabel=r'Relative Error Magnitude', scale_legend=True)
    fig.savefig('./Plot/HeatmapCompEff.pdf', bbox_inches='tight')

### Sweep over removed bias functions ###
if config=='PolyRemoveTFC':
    poly_removes=list(range(-1,10, 1))
    TrainingDf, TrainingStats = train_models(poly_removes=poly_removes)

    fig = plt.figure(figsize=set_size(483.69687*1.05, 1, subplots = (2,1)))
    ax = plt.gca()
    ax.semilogy(poly_removes, TrainingDf['Loss'], 'b+', markersize=15, markeredgewidth=0.01)
    ax.grid(which='major', color='#DDDDDD', linewidth=0.8)
    format_axes(ax=ax, fontsize=20, xlabel = 'Polynomial Order removed', ylabel=r'Relative Error Magnitude', scale_legend=True, force_ticks=poly_removes)
    fig.savefig('./Plot/poly_removes.pdf', bbox_inches='tight')

### Sweep over number of training points ###
if config=='PointsTFC':
    points=list(range(1,200, 1))
    TrainingDf, TrainingStats = train_models(points=points)
    
    fig = plt.figure(figsize=set_size(483.69687*1.05, 1, subplots = (2,1)))
    ax = plt.gca()
    ax.semilogy(points, TrainingDf['Loss'], 'b+', markersize=5 , markeredgewidth=0.01)
    ax.grid(which='major', color='#DDDDDD', linewidth=0.8)
    format_axes(ax=ax, fontsize=20, xlabel = 'Number of Training Points ', ylabel=r'Relative Error Magnitude', scale_legend=True)
    fig.savefig('./Plot/Points.pdf', bbox_inches='tight')


############ X-TFC ###############################

### Sweep over polynomial orders ###
if config=='PolyOrderXTFC':
    poly_orders=list(range(1,200, 2))
    TrainingDf, TrainingStats = train_models(points=[100], poly_orders=poly_orders, poly_removes=[-1], basis_funcs=['ELMTanh'], methods = ["lstsq"])

    fig = plt.figure(figsize=set_size(483.69687*1.05, 1, subplots = (2,1)))
    ax = plt.gca()
    ax.semilogy(poly_orders, TrainingDf['Loss'], 'b+', markersize=5, markeredgewidth=0.01)
    ax.grid(which='major', color='#DDDDDD', linewidth=0.8)
    format_axes(ax=ax, fontsize=20, xlabel = 'Number of Neurons', ylabel=r'Relative Error Magnitude', scale_legend=True)
    fig.savefig('./Plot/NumNeurons.pdf', bbox_inches='tight')

### Sweep over removed bias functions ###
if config=='PolyRemoveXTFC':
    poly_removes=list(range(-1,10, 1))
    TrainingDf, TrainingStats = train_models(points=[100], poly_orders=[50], poly_removes=poly_removes, basis_funcs=['ELMTanh'], methods = ["lstsq"])

    fig = plt.figure(figsize=set_size(483.69687*1.05, 1, subplots = (2,1)))
    ax = plt.gca()
    ax.semilogy(poly_removes, TrainingDf['Loss'], 'b+', markersize=15, markeredgewidth=0.01)
    ax.grid(which='major', color='#DDDDDD', linewidth=0.8)
    format_axes(ax=ax, fontsize=20, xlabel = 'Polynomial Order removed', ylabel=r'Relative Error Magnitude', scale_legend=True, force_ticks=poly_removes)
    fig.savefig('./Plot/poly_removesXTFC.pdf', bbox_inches='tight')

### Sweep over number of training points ###
if config=='PointsXTFC':
    points=list(range(1,200, 1))
    TrainingDf, TrainingStats = train_models(points=points, poly_orders=[50], poly_removes=[-1], basis_funcs=['ELMTanh'], methods = ["lstsq"])
    
    fig = plt.figure(figsize=set_size(483.69687*1.05, 1, subplots = (2,1)))
    ax = plt.gca()
    ax.semilogy(points, TrainingDf['Loss'], 'b+', markersize=5 , markeredgewidth=0.01)
    ax.grid(which='major', color='#DDDDDD', linewidth=0.8)
    format_axes(ax=ax, fontsize=20, xlabel = 'Number of Training Points ', ylabel=r'Relative Error Magnitude', scale_legend=True)
    fig.savefig('./Plot/PointsXTFC.pdf', bbox_inches='tight')


