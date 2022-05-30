#!/usr/bin/env python
# coding: utf-8

import warnings
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=UserWarning)

import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import pykep as pk
import deepxde as dde

from scipy.integrate import odeint
from icecream import ic
from mpl_toolkits.mplot3d import Axes3D
from numpy.linalg import norm

import sys

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

# ## Example Section 
# ## 1D Gravity Trajectory Testing
if RunType =='1D':
    def plot1D_grav(u0, v0, ub, ax=None): 
        x0 = u0
        vx0 = v0

        # mu = pk.MU_SUN/pk.AU**3 * deltat**2

        def f(state, t):
            # x, dx, y, dy = state  # Unpack the state vector
            x, dx, = state  # Unpack the state vector
            r = x #+pk.EARTH_RADIUS
            return dx, -pk.MU_SUN/(r**2) # Derivatives

        state0 = [x0, vx0]
        # state0 = [x0, vx0, y0, vy0]
        t = np.arange(0.0, ub+ub/500, ub/500)

        states = odeint(f, state0, t)   

        if ax==None:
            ic()
            fig = plt.figure(figsize=(5,5))
            ax = plt.gca()
        ax.plot(t, states[:, 0]*1.0, color='green', label = 'True', linewidth=5)
        ax.legend(fontsize = 40)
        ax.ticklabel_format(useOffset=False)
        # ax.view_init(0,0)

        return states


    # In[64]:

    _r0 = 1.5*pk.AU
    _deltat = 1600*24*3600
    v0 = 30*1000

    states = plot1D_grav(_r0, v0, _deltat)

    _rf = states[-1,0]

    nc = _r0
    tnc = np.sqrt(nc**3/pk.MU_SUN)

    # r0 = [_r0/nc,_r0/nc]
    # rf = [_rf/nc, _rf/nc]
    # mu = pk.MU_EARTH/(nc)**3 * tnc**2
    # _r = _r0+pk.EARTH_RADIUS
    # ic(mu*(_r/(nc))/((_r/nc)**2)**(3/2))
    # ic(mu)

    r0 = [_r0/nc]
    rf = [_rf/nc,]   
    mu = pk.MU_SUN/(nc)**3 * tnc**2
    _r = _r0
    ic(mu*(_r/(nc))/((_r/nc)**2)**(3/2))
    ic(_deltat/tnc)
    ic(mu)



    # In[65]:


    """Backend supported: tensorflow.compat.v1, tensorflow, pytorch

    Documentation: https://deepxde.readthedocs.io/en/latest/demos/lorenz.inverse.html
    """

    lb, ub = 0, _deltat/tnc

    def gen_traindata():
        # tsample = np.linspace(0,4,100).reshape(-1,1)
        tsample = np.array(ub).reshape(-1,1)
        return tsample, np.hstack([rf[0]]).reshape(-1,1)
        # return tsample, np.hstack([np.sin(tsample), np.cos(tsample)])

    def Lorenz_system(x, y):
        """Lorenz system.
        dy1/dx = 10 * (y2 - y1)
        dy2/dx = y1 * (15 - y3) - y2
        """ 
        y1, y2 = y[:, 0:1], y[:, 0:1]
        r = y1#+pk.EARTH_RADIUS/nc
        # r = y1+4.263519e-5
        dy1_xx = dde.grad.hessian(y, x)
        return dy1_xx + mu/(r**2)

    np.random.seed()
    tf.compat.v1.reset_default_graph()
    n = np.random.randint(0,10000)
    # n = 878
    n = 6451
    print(f'seed used: {n}')
    dde.config.set_random_seed(n)


    def boundary(_, on_initial):
        return on_initial


    geom = dde.geometry.TimeDomain(lb, ub)

    # Initial conditions
    ic1 = dde.IC(geom, lambda X: r0[0], boundary)
    # Get the train data
    observe_t, ob_y = gen_traindata()
    observe_y0 = dde.PointSetBC(observe_t, ob_y[:, 0:1], component=0)

    data = dde.data.PDE(
        geom,
        Lorenz_system,
        [ic1, observe_y0],
        num_domain=200,
        num_boundary=20,
        anchors=observe_t,
    )

    net = dde.maps.FNN([1] + [50] * 3 + [1], "sigmoid", "Glorot uniform")
    net.apply_output_transform(
        lambda x, y: (ub-x)/ub*r0 + (x/ub)*rf + x/ub*(ub-x)/ub*y
    )
    model = dde.Model(data, net)


    # lr = tf.keras.optimizers.schedules.PiecewiseConstantDecay([3000.0, 5000.0, 10000.0 ],[1e-2, 5e-3, 1e-3, 1e-4])
    # lr = 0.1
    # decay weigth = 0.5 : decrease by 50% every 2000 steps
    model.compile("adam", lr=0.01, 
                decay = ("inverse time", 10000, 0.5)
    )

    losshistory, train_state = model.train(epochs=300, display_every=1000)


    # dde.config.set_default_float("float32") #If L-BFGS stops earlier than expected, set the default float type to ‘float64’:If L-BFGS stops earlier than expected, set the default float type to ‘float64’:
    model.compile("L-BFGS-B")
    dde.optimizers.config.set_LBFGS_options(maxcor=100, ftol=0, gtol=1e-10, maxiter=15000, maxfun=None, maxls=50)
    losshistory, train_state = model.train(display_every=1000)

    GradCallback = dde.callbacks.FirstDerivative(0)


    # In[66]:


    import matplotlib.pyplot as plt

    plot_trajec = 0
    ax = dde.saveplot(losshistory, train_state, issave=False, isplot=True)
    fig = plt.figure(figsize=(10,7))
    ax = fig.add_subplot(111)

    tsample = np.linspace(lb, ub ,30).reshape(-1,1)
    upred = model.predict(tsample)[:,plot_trajec]*nc
    ax.plot((tsample*tnc), upred, label = 'predicted',color='red', marker='o', markersize=10)
    ax.ticklabel_format(useOffset=False)


    X_train, y_train = train_state.X_train, train_state.y_train
    X_test, y_test, best_y, best_ystd = train_state.X_test, train_state.y_test, train_state.best_y, train_state.best_ystd
    idx = np.argsort(X_test[:, 0])  
    X = X_test[idx, 0]
    best_y = best_y[idx]
    plt.plot(X*tnc, best_y*nc,  label = 'predicted', color='red', marker='o', markersize=5)

    ax.set_xlabel('Time', fontsize=30)
    ax.set_ylabel('h [m]', fontsize=30)
    ax.tick_params(labelsize=40)
    ax.legend(fontsize=40)

    # ax.set_ylim([-2,2]
    # ax.set_xlim([-2,2])
    def dydx(x, y):
        return dde.grad.jacobian(y, x, i=plot_trajec, j=0)
    dy_dx = model.predict(np.array([[0]]), operator=dydx)

    print(f"V0 Desired: {v0} m/s \nV0 Predicted: {dy_dx*nc/tnc} m/s \nError : {100 - v0/(dy_dx*nc/tnc)*100} %")

    print(f'Intial Accuracy \n\t Predicted:{best_y[0]} \n\t Desired: {r0[0]} \n\t % Error: {100 - best_y[0]/r0[0]*100}')
    print(f'Final Accuracy \n\t Predicted:{best_y[-1]} \n\t Desired: {rf[0]} \n\t % Error: {100 - best_y[-1]/rf[0]*100}')

    TrueStates = plot1D_grav(r0[plot_trajec]*nc, v0, ub*tnc, ax=ax)

    # ax.view_init(90,0)
    # plt.show()

    # ax = pk.orbit_plots.plot_lambert(lamsol_list, N=60, sol=0, units=pk.AU, legend=False, axes=None, alpha=1.)  
    # ax.view_init(90,0)


    # In[6]:


    def props(cls):   
        return [i for i in cls.__dict__.keys()]
    
    properties = props(train_state)


# ## 3D Gravity Trajectory Testing
# In[17]:
def plot3D_grav(u0, v0, ub, uf=[None,None,None], mode='SUN', ax=None): 
        x0, y0, z0 = u0
        vx0, vy0, vz0 = v0
        fontsize = 30

        # mu = pk.MU_SUN/pk.AU**3 * deltat**2
        if mode=='SUN':
            def f(state, t):
                # x, dx, y, dy = state  # Unpack the state vector
                x, dx, y, dy, z, dz = state  # Unpack the state vector
                rx = x
                ry = y
                rz = z
                return dx, -pk.MU_SUN*rx/(rx**2 + ry**2 + rz**2)**(3/2), \
                    dy, -pk.MU_SUN*ry/(rx**2 + ry**2 + rz**2)**(3/2), \
                    dz, -pk.MU_SUN*rz/(rx**2 + ry**2 + rz**2)**(3/2) # Derivatives
                
        elif mode=='EARTH':
            def f(state, t):
                # x, dx, y, dy = state  # Unpack the state vector
                x, dx, y, dy = state  # Unpack the state vector
                rx = x + pk.EARTH_RADIUS
                ry = y + pk.EARTH_RADIUS
                return dx, -pk.MU_EARTH*rx/(rx**2 + ry**2)**(3/2), dy, -pk.MU_EARTH*ry/(rx**2 + ry**2)**(3/2) # Derivatives

        #state0 = [u0, 0]
        state0 = [x0, vx0, y0, vy0, z0, vz0]
        t = np.arange(0.0, ub+ub/50000, ub/50000)

        states = odeint(f, state0, t)

        if ax==None:
            # ic()
            fig = plt.figure(figsize=(10,5))
            ax = plt.gca()

        ax.ticklabel_format(useOffset=False)
        ax.plot(states[:, 0]/1000, states[:, 2]/1000, color='green', label = 'True', linewidth=2)
        ax.plot(states[0, 0]/1000, states[0, 2]/1000, color='blue', label = 'Initial Position', marker='o', markersize=10)
        # ax.plot(states[-1, 0], states[-1, 2], color='red', label = 'Final Position', marker='o', markersize=10)
        if all(uf):
            ax.plot([0, states[0,0]/1000], [0, states[0,2]/1000], '-r')
            ax.plot([0, uf[0]/1000], [0, uf[1]/1000], '-r')
            ax.plot(uf[0]/1000, uf[1]/1000, color='orange', label = 'Lambert Position', marker='o', markersize=10)

        ax.legend(fontsize = fontsize)
        ax.xaxis.offsetText.set_fontsize(fontsize)
        ax.yaxis.offsetText.set_fontsize(fontsize)
        plt.xticks(fontsize= fontsize)
        plt.yticks(fontsize= fontsize)

        plt.show()
        # ic(states[-1,:])
        # ax.view_init(0,0)

        return states

if RunType=='3D':
   

    # In[14]:


    class LambertEq():
        def __init__(self, callback_config):
            # Reset workspace 
            np.random.seed()
            tf.compat.v1.reset_default_graph()
            self.callback_config = callback_config
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


    # In[81]:


    class LambertEq(LambertEq):
        def Get_Lambert(self, new=True, shortway=True, defined=False):
            
            ############################################################################
            ########################## Define Lambert Problem ##########################
            ############################################################################

            if new and defined:
                # Set limits on TOF and starting date
                Start_Epoch     = int(pk.epoch_from_string("2001-01-01 00:00:00").mjd)
                End_Epoch       = int(pk.epoch_from_string("2001-09-01 00:00:00").mjd)
                
                # specify target planet
                _target = 'mars'
                
                self.t1 = np.array([Start_Epoch])
                self.t2 = End_Epoch
                self._TOF = self.t2-self.t1

            else:

                # Set limits on TOF and starting date
                Start_Epoch     = int(pk.epoch_from_string("2001-01-01 00:00:00").mjd)
                End_Epoch       = int(pk.epoch_from_string("2031-01-01 00:00:00").mjd)
                TOF_range       = [10, 300]
                
                # specify target planet
                _target = 'mars'

                # only generate new problem if required
                if new:
                    self.t1              = np.random.randint(low=Start_Epoch, high=End_Epoch, size=1)
                    self._TOF             = np.random.randint(low=TOF_range[0], high=TOF_range[1], size=1)
                    self.t2              = self.t1 + self._TOF

            # Get Ephemeris data from pykep
            Departure       = pk.planet.jpl_lp('earth') 
            Target          = pk.planet.jpl_lp(_target)

            States0         = Departure.eph(pk.epoch(int(self.t1), 'mjd'))
            Statesf         = Target.eph(pk.epoch(int(self.t2), 'mjd'))

            self._r0              = np.array(States0[0])
            self._rf              = np.array(Statesf[0])

            if shortway:
                clockwise = True if np.cross(self._r0,self._rf)[2] < 0 else False
            else:
                clockwise = True if np.cross(self._r0,self._rf)[2] >= 0 else False


            ################################################################################
            ############################# Solve Lambert Problem ############################
            ################################################################################

            lamsol_list = pk.lambert_problem(r1=self._r0, r2=self._rf, tof=int(self._TOF)*24*3600,
                                            mu=pk.MU_SUN, cw=clockwise)
            self.v1 = lamsol_list.get_v1()[0]
            v2 = lamsol_list.get_v2()[0]
                                
            self.TOF = self.t2-self.t1
            ################################################################################
            ############################# Set up training data #############################
            ################################################################################


            # Define normamlising constants for distance and time
            self.nc = np.linalg.norm(self._r0)
            self.tnc = np.sqrt(self.nc**3/pk.MU_SUN)

            # Normalise input data
            self.r0 = self._r0/self.nc
            self.rf = self._rf/self.nc
            self.mu = pk.MU_SUN/(self.nc)**3 * self.tnc**2

            # Specifiy short or long way solution (dtheta<180?)
            self.short_way=shortway
            ic(self.TOF*24*3600/self.tnc)

            ################################################################################
            ############################# Ensure solution exists ###########################
            ################################################################################


            c = norm(self._r0 - self._rf)
            s = (norm(self._r0) +  norm(self._rf) + c) /2

            alpha_m = np.pi
            beta_m = 2*np.arcsin(np.sqrt((s-c)/s))

            # Minimum Energy Solution - determines long or short time solution 
            dt_m = np.sqrt(s**3/(8*pk.MU_SUN))*(np.pi - beta_m + np.sin(beta_m))     
            dtheta = np.arccos(np.dot(self._r0,self._rf)/(norm(self._r0)*norm(self._rf)))

            # if long way specified, adjust change in true anomaly for parabolic transfer time calculation
            if not self.short_way:
                print('Adjusting for long way solution')
                dtheta = 2*np.pi - dtheta
            # parabolic transfer time - minimum 
            dt_p = np.sqrt(2)/3*(s**1.5 - np.sign(np.sin(dtheta))*(s-c)**1.5)/np.sqrt(pk.MU_SUN)

            # Determine if desired solution corresponds to short or long time solution
            if self.TOF[0] < dt_m/3600/24 :
                self.short_time = True
            else:
                self.short_time = False
                
            # If desired solution non existent generate new solution with identical parameters
            if self.TOF[0] < dt_p/3600/24:
                print('invlaid TOF, generating new problem')
                self.Get_Lambert(shortway = self.short_way)

                # Avoids double execution of function
                return 0
            ############################################################################
            ############################# Get Correct SMA ##############################
            ############################################################################

            if new:
                print(self.color.BOLD + self.color.GREEN + 
                    f'Start Date: {pk.epoch(int(self.t1), "mjd")}')
                print(f'End Date:   {pk.epoch(int(self.t2), "mjd")}\n' + self.color.RESET)

                print(f'Min E deltaT: {dt_m/3600/24:.3f} days')
                print(f'parabolic deltaT: {dt_p/3600/24:.3f} days')
                print(f'Desired TOF:  {self.TOF[0]} days\n')

            
            
            _ = plot3D_grav(self._r0, self.v1, self.TOF*24*3600, self._rf)

            ############################################################################
            ############################ Fill in Config Dict ###########################
            ############################################################################
            self.callback_config['Lambert Problem']['Start Date'] = pk.epoch(int(self.t1), "mjd")
            self.callback_config['Lambert Problem']['End Date'] = pk.epoch(int(self.t2), "mjd")
            self.callback_config['Lambert Problem']['TOF'] = self.TOF[0]


    # In[89]:


    class LambertEq(LambertEq):
        def TrainModel(self):

            ############################################################################
            ############################# Set Seed for reproducibility #################
            ############################################################################
            
            np.random.seed()
            tf.compat.v1.reset_default_graph()
            n = np.random.randint(0,10000)
            n = 878
            callback_config['seed'] = n
            print(f'seed used: {n}')
            dde.config.set_random_seed(n)

            ############################################################################
            ############################# Set up Model config options ##################
            ############################################################################
            self._deltat = self.TOF*24*3600
            self.lb, self.ub = 0, self._deltat/self.tnc

            def gen_traindata():
                tsample = np.array(self.ub).reshape(-1,1)
                return tsample, np.hstack(  [self.rf[0], self.rf[1], self.rf[2]]  ).reshape(-1,3)

            def ODE_system(x, y):

                rx, ry, rz = y[:, 0], y[:, 1], y[:,2]
                dy1_xx = dde.grad.hessian(y, x, component=0)
                dy2_xx = dde.grad.hessian(y, x, component=1)
                dy3_xx = dde.grad.hessian(y, x, component=2)

                return [
                    dy1_xx + self.mu*rx/((rx**2) + (ry**2) + (rz**2))**(3/2),
                    dy2_xx + self.mu*ry/((rx**2) + (ry**2) + (rz**2))**(3/2),
                    dy3_xx + self.mu*rz/((rx**2) + (ry**2) + (rz**2))**(3/2),
                ]

            def boundary(_, on_initial):
                return on_initial

            geom = dde.geometry.TimeDomain(self.lb, self.ub)

            # Initial conditions
            ic1 = dde.IC(geom, lambda X: self.r0[0], boundary, component=0)
            ic2 = dde.IC(geom, lambda X: self.r0[1], boundary, component=1)
            ic3 = dde.IC(geom, lambda X: self.r0[2], boundary, component=2)
            # ic3 = dde.IC(geom, lambda X: r0[2], boundary, component=2)

            # Get the train data
            observe_t, ob_y = gen_traindata()
            observe_y0 = dde.PointSetBC(observe_t, ob_y[:, 0:1], component=0)
            observe_y1 = dde.PointSetBC(observe_t, ob_y[:, 1:2], component=1)
            observe_y2 = dde.PointSetBC(observe_t, ob_y[:, 2:3], component=2)
            # observe_y2 = dde.PointSetBC(observe_t, ob_y[:, 2:3], component=2)

            data = dde.data.PDE(
                geom,
                ODE_system,
                [ic1, ic2, ic3, observe_y0, observe_y1, observe_y2],
                num_domain=1000,
                num_boundary=2,
                anchors=observe_t,
            )

            layers = callback_config['Model Architecture']['Layers']
            neurons =   callback_config['Model Architecture']['Neurons p Layer']
            net = dde.maps.FNN([1] + [neurons] * layers + [3], "tanh", "Glorot uniform")

            net.apply_output_transform(
            lambda x, y: (self.ub-x)/self.ub*self.r0 + (x/self.ub)*self.rf + x*(self.ub-x)*y
            )

            self.model = dde.Model(data, net)

            ############################################################################
            ################################# Train Model ##############################
            ############################################################################
            
            # decay weigth = 0.5 : decrease by 50% every 2000 steps
            self.model.compile("adam", lr=callback_config['lr'], 
                        decay = ("inverse time", 
                                callback_config['Reduce lr']["steps"], 
                                callback_config['Reduce lr']["factor"]),
                                loss_weights = [1, 1, 1,  1, 1, 1,  1, 1, 1]
            )
            self.losshistory, self.train_state = self.model.train(epochs=callback_config['Adam Epochs'], display_every=100)

            # dde.config.set_default_float("float64") #If L-BFGS stops earlier than expected, set the default float type to ‘float64’:If L-BFGS stops earlier than expected, set the default float type to ‘float64’:
            self.model.compile("L-BFGS-B")
            dde.optimizers.config.set_LBFGS_options(maxcor=100, ftol=0, gtol=1e-10, maxiter=15000, maxfun=None, maxls=50)
            self.losshistory, self.train_state = self.model.train(display_every=100)


    # In[46]:


    class LambertEq(LambertEq):
        def Get_Error(self):
            tsample = np.linspace(self.lb, self.ub ,30).reshape(-1,1)
            upred = self.model.predict(tsample)*self.nc
            ic(upred[0,:])
            ic(upred[-1,:])

            print(f'Intial X Accuracy \n\t Predicted:{upred[0,0]/self.nc} \n\t Desired: {self.r0[0]} \n\t % Error: {100 - upred[0,0]/self.nc/self.r0[0]*100}')
            print(f'Intial Y Accuracy \n\t Predicted:{upred[0,1]/self.nc} \n\t Desired: {self.r0[1]} \n\t % Error: {100 - upred[0,1]/self.nc/self.r0[1]*100}')
            print(f'Intial Z Accuracy \n\t Predicted:{upred[0,2]/self.nc} \n\t Desired: {self.r0[2]} \n\t % Error: {100 - upred[0,2]/self.nc/self.r0[2]*100}')

            print(f'Final X Accuracy \n\t Predicted:{upred[-1,0]/self.nc} \n\t Desired: {self.rf[0]} \n\t % Error: {100 - upred[-1,0]/self.nc/self.rf[0]*100}')
            print(f'Final Y Accuracy \n\t Predicted:{upred[-1,1]/self.nc} \n\t Desired: {self.rf[1]} \n\t % Error: {100 - upred[-1,1]/self.nc/self.rf[1]*100}')
            print(f'Final Z Accuracy \n\t Predicted:{upred[-1,2]/self.nc} \n\t Desired: {self.rf[2]} \n\t % Error: {100 - upred[-1,2]/self.nc/self.rf[2]*100}')

            import matplotlib.pyplot as plt

            ax = dde.saveplot(self.losshistory, self.train_state, issave=False, isplot=True)
            fig = plt.figure(figsize=(30,20))
            ax = fig.add_subplot(111)

            ## Last Model
            ax.plot(upred[:,0]/1000, upred[:,1]/1000, label = 'predicted',color='red', marker='o', markersize=10) 
            ax.ticklabel_format(useOffset=False)

            ## Best Model

            X_train, y_train = self.train_state.X_train, self.train_state.y_train
            X_test, y_test, best_y, best_ystd = self.train_state.X_test, self.train_state.y_test, self.train_state.best_y, self.train_state.best_ystd
            idx = np.argsort(X_test[:, 0])
            X = X_test[idx, 0]
            Pred_x = best_y[idx, 0]
            Pred_y = best_y[idx, 1]
            # ic(Pred_y)
            plt.plot(Pred_x*self.nc/1000, Pred_y*self.nc/1000, "--r", label = 'predicted',color='red', marker='o', markersize=5)

            ## Correct Solution
            ax.set_xlabel('X [m]', fontsize=30)
            ax.set_ylabel('Y [m]', fontsize=30)
            ax.tick_params(labelsize=40)
            ax.legend(fontsize=40)

            # ax.set_ylim([-2,2]
            # ax.set_xlim([-2,2])

            def dydx1(x, y):
                return dde.grad.jacobian(y, x, i=0, j=0)

            def dydx2(x, y):
                return dde.grad.jacobian(y, x, i=1, j=0)

            def dydx3(x, y):
                return dde.grad.jacobian(y, x, i=2, j=0)
            

            self.dy_dx1 = self.model.predict(np.array([[0]]), operator=dydx1)[0][0]
            self.dy_dx2 = self.model.predict(np.array([[0]]), operator=dydx2)[0][0]
            self.dy_dx3 = self.model.predict(np.array([[0]]), operator=dydx3)[0][0]

            error = (np.array([self.dy_dx1, self.dy_dx2, self.dy_dx3]) - self.v1/self.nc*self.tnc) / (self.v1/self.nc*self.tnc)

            print(f'predicted velocity: {self.dy_dx1, self.dy_dx2, self.dy_dx3}. \nDesired Velocity:{self.v1/self.nc*self.tnc}\n')
            print(f'% Error: {error*100}')

            def dydxx1(x, y):
                return dde.grad.hessian(y, x, component=0)

            

            # dy_dxx = self.model.predict(np.array([[0]]), operator=dydxx1)
            # ic(mu*r0[0]/((self.r0[0]**2) + (self.r0[1]**2) + self.r0[2]**2)**(3/2) , dy_dxx)

            # ic(best_y.shape)

            TrueStates =plot3D_grav(self._r0, self.v1, self.TOF*24*3600, self._rf, ax=ax)

            # ax.view_init(90,0)
            # plt.show()

            # ax = pk.orbit_plots.plot_lambert(lamsol_list, N=60, sol=0, units=pk.AU, legend=False, axes=None, alpha=1.)  
            # ax.view_init(90,0)


    # In[90]:


    tf.keras.backend.clear_session()

    callback_config = {}
    callback_config['seed'] = {}
    callback_config['Lambert Problem'] = {}
    callback_config['Reduce lr'] = {}
    callback_config['Model Architecture'] = {}

    callback_config['lr'] = 0.01
    callback_config['Adam Epochs'] = 100

    callback_config['Reduce lr']["factor"] = 0.5
    callback_config['Reduce lr']["steps"] = 10000



    callback_config['Lambert Problem']['Start Date'] = {}
    callback_config['Lambert Problem']['End Date'] = {}
    callback_config['Lambert Problem']['TOF'] = {}

    callback_config['Model Architecture']['Layers'] = 5
    callback_config['Model Architecture']['Neurons p Layer'] = 1000




    # In[91]:


    l = LambertEq(callback_config)
    l.Get_Lambert(shortway=True, defined=True)
    l.TrainModel()
    l.Get_Error()


# # Non Class IMplementation

# In[212]:

if RunType =='2D':
    _r0 = [-30*0.181359*pk.AU, 0.5*0.966435*pk.AU, 0]
    _deltat = 300*24*3600
    # v0 = [30114, 11660, 0]
    v0 = [-1004, 800, 0]   

    # _r0 = [pk.AU, 0]
    # _deltat = 100*24*3600
    # v0 = [20000, 20000]

    states = plot3D_grav(_r0, v0, _deltat, mode='SUN')
    _rf = states[-1,[0,2]]
    states = plot3D_grav(_r0, v0, _deltat, mode='SUN', uf=_rf)
        

    nc = norm(_r0)
    tnc = np.sqrt(nc**3/pk.MU_SUN)
    # tnc = _deltat

    r0 = [_r0[0]/nc,_r0[1]/nc]
    rf = [_rf[0]/nc, _rf[1]/nc]
    ic(rf)

    # mu = pk.MU_EARTH/(nc)**3 * tnc**2
    # _r = _r0[0] + pk.EARTH_RADIUS, _r0[1] + pk.EARTH_RADIUS
    # ic(mu*(_r[0]/(nc))/((_r[0]/nc)**2 + (_r[1]/nc)**2)**(3/2))
    # ic(mu*(_r[1]/(nc))/((_r[0]/nc)**2 + (_r[1]/nc)**2)**(3/2))
    # ic(mu)

    mu = pk.MU_SUN/(nc)**3 * tnc**2
    _r = _r0
    ic(mu*(_r[0]/(nc))/((_r[0]/nc)**2 +(_r[1]/nc)**2)**(3/2))
    ic(mu*(_r[1]/(nc))/((_r[0]/nc)**2 + (_r[1]/nc)**2)**(3/2))
    ic(_deltat/tnc)


    # In[205]:


    """Backend supported: tensorflow.compat.v1, tensorflow, pytorch

    Documentation: https://deepxde.readthedocs.io/en/latest/demos/lorenz.inverse.html
    """

    lb, ub = 0, _deltat/tnc

    def gen_traindata():
        # tsample = np.linspace(0,4,100).reshape(-1,1)
        tsample = np.array(ub).reshape(-1,1)
        return tsample, np.hstack(  [rf[0], rf[1]]  ).reshape(-1,2)
        # return tsample, np.hstack([np.sin(tsample), np.cos(tsample)])

    def Lorenz_system(x, y):
        """Lorenz system.
        dy1/dx = 10 * (y2 - y1)
        dy2/dx = y1 * (15 - y3) - y2
        """ 
        rx, ry = y[:, 0], y[:, 1]
        # r = y1+pk.EARTH_RADIUS/nc

        dy1_xx = dde.grad.hessian(y, x, component=0)
        dy2_xx = dde.grad.hessian(y, x, component=1)

        return [
            dy1_xx + mu*rx/((rx**2) + (ry**2))**(3/2),
            dy2_xx + mu*ry/((rx**2) + (ry**2))**(3/2),
            # dy3_xx*(y1**2 + y2**2 + y3**2)**(3/2) + mu*y3
        ]

    np.random.seed()
    tf.compat.v1.reset_default_graph()
    n = np.random.randint(0,10000)
    # n = 878
    n = 7559
    print(f'seed used: {n}')
    dde.config.set_random_seed(n)

    def boundary(_, on_initial):
        return on_initial


    timedomain = dde.geometry.TimeDomain(lb, ub)

    # Initial conditions
    ic1 = dde.IC(timedomain, lambda X: r0[0], boundary, component=0)
    ic2 = dde.IC(timedomain, lambda X: r0[1], boundary, component=1)
    # ic3 = dde.IC(geom, lambda X: r0[2], boundary, component=2)
    # Get the train data
    observe_t, ob_y = gen_traindata()
    observe_y0 = dde.PointSetBC(observe_t, ob_y[:, 0:1], component=0)
    observe_y1 = dde.PointSetBC(observe_t, ob_y[:, 1:2], component=1)
    # observe_y2 = dde.PointSetBC(observe_t, ob_y[:, 2:3], component=2)

    data = dde.data.TimePDE(
        timedomain,
        Lorenz_system,
        [ic1, ic2, observe_y0, observe_y1],
        num_domain=1000,
        num_boundary=2,
        anchors=observe_t,
    )

    activation = f"LAAF-{10} tanh"  # "LAAF-10 relu"
    net = dde.maps.FNN([1] + [100]*5 + [2], activation, "Glorot uniform")
    # hard initial conditions. 
    # see https://deepxde.readthedocs.io/en/latest/demos/pinn_forward/diffusion.1d.exactBC.html#:~:text=Then%20we%20construct,both%20hard%20conditions.
    net.apply_output_transform(
        lambda x, y: (ub-x)/ub*r0 + (x/ub)*rf + x/ub*(ub-x)/ub*y
    )
    model = dde.Model(data, net)



    # decay weigth = 0.5 : decrease by 50% every 2000 steps
    model.compile("adam", lr=0.01, 
                decay = ("inverse time", 10000, 0.5),
                loss_weights = [1, 1, 1, 1, 1, 1]
    )
    losshistory, train_state = model.train(epochs=100, display_every=100)

    # dde.config.set_default_float("float64") #If L-BFGS stops earlier than expected, set the default float type to ‘float64’:If L-BFGS stops earlier than expected, set the default float type to ‘float64’:
    model.compile("L-BFGS-B", 
                loss_weights = [1, 1, 1, 1, 1, 1])
    dde.optimizers.config.set_LBFGS_options(maxcor=100, ftol=0, gtol=1e-10, maxiter=15000, maxfun=None, maxls=50)
    losshistory, train_state = model.train(display_every=100)


    # In[206]:


    tsample = np.linspace(lb, ub ,30).reshape(-1,1)
    upred = model.predict(tsample)*nc
    ic(upred[0,:])
    ic(upred[-1,:])

    print(f'Intial X Accuracy \n\t Predicted:{upred[0,0]/nc} \n\t Desired: {r0[0]} \n\t % Error: {100 - upred[0,0]/nc/r0[0]*100}')
    print(f'Intial Y Accuracy \n\t Predicted:~{upred[0,1]/nc} \n\t Desired: {r0[1]} \n\t % Error: {100 - upred[0,1]/nc/(r0[1])*100}')

    print(f'Final X Accuracy \n\t Predicted:{upred[-1,0]/nc} \n\t Desired: {rf[0]} \n\t % Error: {100 - upred[-1,0]/nc/rf[0]*100}')
    print(f'Final Y Accuracy \n\t Predicted:{upred[-1,1]/nc} \n\t Desired: {rf[1]} \n\t % Error: {100 - upred[-1,1]/nc/rf[1]*100}')

    def dydx1(x, y):
        return dde.grad.jacobian(y, x, i=0, j=0)

    def dydx2(x, y):
        return dde.grad.jacobian(y, x, i=1, j=0)

    def dydx3(x, y):
        return dde.grad.jacobian(y, x, i=2, j=0)


    dy_dx1 = model.predict(np.array([[0]]), operator=dydx1)[0][0]
    dy_dx2 = model.predict(np.array([[0]]), operator=dydx2)[0][0]

    v1 = v0[0:2]
    error = (np.array([dy_dx1, dy_dx2]) - v1/nc*tnc) / (v1/nc*tnc)

    print(f'predicted velocity: {dy_dx1, dy_dx2}. \nDesired Velocity:{v1/nc*tnc}\n')
    print(f'% Error: {error*100}')


    # In[211]:


    import matplotlib.pyplot as plt

    plot_trajec = 0

    ax = dde.saveplot(losshistory, train_state, issave=False, isplot=True)
    fig = plt.figure(figsize=(15,10))
    ax = fig.add_subplot(111)

    ## Last Model
    ax.plot(upred[:,0]/1000, upred[:,1]/1000, label = 'predicted',color='red', marker='o', markersize=10)
    ax.ticklabel_format(useOffset=False)

    ## Best Model

    X_train, y_train = train_state.X_train, train_state.y_train
    X_test, y_test, best_y, best_ystd = train_state.X_test, train_state.y_test, train_state.best_y, train_state.best_ystd
    idx = np.argsort(X_test[:, 0])
    X = X_test[idx, 0]
    Pred_x = best_y[idx, 0]
    Pred_y = best_y[idx, 1]
    # ic(Pred_y)
    plt.plot(Pred_x*nc/1000, Pred_y*nc/1000, "--r", label = 'predicted',color='red', marker='o', markersize=5)

    ## Correct Solution
    ax.set_xlabel('X [km]', fontsize=30)
    ax.set_ylabel('Y [km]', fontsize=30)
    ax.tick_params(labelsize=40)
    ax.legend(fontsize=40)

    # ax.set_ylim([-2,2]
    # ax.set_xlim([-2,2])
    def dydx1(x, y):
        return dde.grad.jacobian(y, x, i=0, j=0)
    dy_dx1 = model.predict(np.array([[0]]), operator=dydx1)

    def dydxx1(x, y):
        return dde.grad.hessian(y, x, component=0)

    dy_dxx = model.predict(np.array([[0]]), operator=dydxx1)
    # ic(mu*r0[0]/((r0[0]**2) + (r0[1]**2))**(3/2) , dy_dxx)

    # ic(best_y.shape)

    TrueStates = plot3D_grav(_r0, v0, ub*tnc, mode='SUN', ax=ax, uf=_rf)
    # ic(upred[-1,1]/1e5)
    # ic(TrueStates[-1,:]*nc)
    # ic(_rf)

    # ax.view_init(90,0)
    # plt.show()

    # ax = pk.orbit_plots.plot_lambert(lamsol_list, N=60, sol=0, units=pk.AU, legend=False, axes=None, alpha=1.)  
    # ax.view_init(90,0)


    # In[190]:


    # def dydxx1(x, y):
    #     return dde.grad.hessian(y, x, component=0)

    # def dydxx2(x, y):
    #     return dde.grad.hessian(y, x, component=1)

    # upred2=upred/nc
    # dy_dxx1 = model.predict(tsample, operator=dydxx1)
    # dy_dx1 = model.predict(np.array(tsample), operator=dydx1)

    # dy_dxx2 = model.predict(np.array([[0]]), operator=dydxx2)[0][0]

    # print(dy_dxx1.T)
    # # print((norm(upred2, axis=1)))
    # print( mu*upred2[:,0]/(norm(upred2, axis=1)**3), '\n')

    # print(dy_dx1.T)
    # print(states[:,1].T/nc*tnc)

    # print(dy_dxx2)
    # print( mu*r0[1]/(norm(r0)**3))


# In[ ]:




