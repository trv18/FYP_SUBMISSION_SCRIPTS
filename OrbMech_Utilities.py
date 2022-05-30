from scipy.integrate  import odeint
import matplotlib.pyplot as plt
import pykep             as pk
from Visualise_Tools import set_size


def plot3D_grav(u0, v0, ub, uf=[None,None,None], mode='SUN', ax=None, J2=0, R=0, return_t_array=False, Include_J2 = False): 
        x0, y0, z0 = u0
        vx0, vy0, vz0 = v0
        fontsize = 12

        # mu = pk.MU_SUN/pk.AU**3 * deltat**2
        if mode=='SUN':
            def f(state, t):
                # x, dx, y, dy = state  # Unpack the state vector
                x, dx, y, dy, z, dz = state  # Unpack the state vector
                rx = x
                ry = y
                rz = z
                return dx, -pk.MU_SUN*rx/(rx**2 + ry**2 + rz**2)**(3/2)* ( 1.0 + Include_J2*1.5*J2*(R**2/(rx**2 + ry**2 + rz**2))*(1-5*rz**2/(rx**2 + ry**2 + rz**2))), \
                       dy, -pk.MU_SUN*ry/(rx**2 + ry**2 + rz**2)**(3/2)* ( 1.0 + Include_J2*1.5*J2*(R**2/(rx**2 + ry**2 + rz**2))*(1-5*rz**2/(rx**2 + ry**2 + rz**2))), \
                       dz, -pk.MU_SUN*rz/(rx**2 + ry**2 + rz**2)**(3/2)* ( 1.0 + Include_J2*1.5*J2*(R**2/(rx**2 + ry**2 + rz**2))*(3-5*rz**2/(rx**2 + ry**2 + rz**2))) # Derivatives
                
        elif mode=='EARTH':
            def f(state, t):
                # x, dx, y, dy = state  # Unpack the state vector   
                x, dx, y, dy = state  # Unpack the state vector
                rx = x + pk.EARTH_RADIUS
                ry = y + pk.EARTH_RADIUS
                return dx, -pk.MU_EARTH*rx/(rx**2 + ry**2)**(3/2), dy, -pk.MU_EARTH*ry/(rx**2 + ry**2)**(3/2) # Derivatives

        #state0 = [u0, 0]
        state0 = [x0, vx0, y0, vy0, z0, vz0]
        t = np.linspace(0.0, ub, 20000)
        states = odeint(f, state0, t)

        if ax==None:
            # ic()
            fig = plt.figure(figsize=set_size(483.69687*1.05, 1))
            ax = plt.gca()

        ax.ticklabel_format(useOffset=False)
        ax.plot(states[:, 0]/1000, states[:, 2]/1000, color='green', label = 'True Solution', linewidth=2)
        ax.plot(states[0, 0]/1000, states[0, 2]/1000, color='blue', label = 'Initial Position', marker='o', markersize=10)

        # ax.plot(states[-1, 0], states[-1, 2], color='red', label = 'Final Position', marker='o', markersize=10)
        if all(uf):
            ax.plot([0, states[0,0]/1000], [0, states[0,2]/1000], '-', color='#EEEEEE')
            ax.plot([0, uf[0]/1000], [0, uf[1]/1000], '-', color='#EEEEEE')
            ax.plot(uf[0]/1000, uf[1]/1000, color='orange', label = 'Lambert Position', marker='o', markersize=10)

        ax.legend(fontsize = fontsize)
        ax.xaxis.offsetText.set_fontsize(fontsize)
        ax.yaxis.offsetText.set_fontsize(fontsize)
        plt.xticks(fontsize= fontsize)
        plt.yticks(fontsize= fontsize)

        plt.show()
        # ic(states[-1,:])
        # ax.view_init(0,0)
        if return_t_array:
            return states, t
        else:
            return states

#------------------------------------------------------------------------
#------------------------------------------------------------------------
#------------------------- Get Position Error ---------------------------
#------------------------------------------------------------------------
#------------------------------------------------------------------------ 

from astropy import units as u
import astropy
import numpy as np
from numpy.linalg import norm

from poliastro.bodies import Earth, Mars, Sun
from poliastro.twobody import Orbit
from poliastro.twobody.propagation import cowell, farnocchia, propagate
from poliastro.maneuver import Maneuver
from poliastro.iod import izzo

def Get_PropagationError():
    with open('OrbitParams.npy', 'rb') as file:
        r0 = np.load(file)* u.m
        rf = np.load(file)/1000*u.km
        v0 = np.load(file)* u.m/u.s
        vf = np.load(file)* u.m/u.s

        v1 = np.load(file)* u.m/u.s
        v1_pred = np.load(file)* u.m/u.s

        TOF = np.load(file)[0]
        t1 = np.load(file)[0]

    # convert mjd to jd
    t1 = astropy.time.Time(t1+2400000.5, format='jd')


    PyKEP_orb = Orbit.from_vectors(Sun, r0, v1, epoch = t1)
    TFC_orb   = Orbit.from_vectors(Sun, r0, v1_pred, epoch = t1)

    PyKEP_prop = PyKEP_orb.propagate(TOF*3600*24*u.second, method=cowell)
    TFC_prop = TFC_orb.propagate(TOF*3600*24*u.second, method=cowell)

    rf_pk = PyKEP_prop.r.to(u.km)
    rf_model = TFC_prop.r.to(u.km)

    pk_error = abs((rf_pk - rf)/rf)
    TFC_error = abs((rf_model - rf)/rf)

    print('PyKEP error in km:' , norm(pk_error*rf))
    print('TFC   error  in km:' , norm(TFC_error*rf), '\n')
    ratio = norm(pk_error*rf)/norm(TFC_error*rf)


    r0_orb =Orbit.from_vectors(Sun, r0.to(u.meter), v0, epoch = t1)
    rf_orb =Orbit.from_vectors(Sun, rf.to(u.meter), vf, epoch = t1 + TOF*u.day)
    sol = Maneuver.lambert(r0_orb, rf_orb, short=True, M=0)
    (vlam, v), = izzo.lambert(Sun.k, r0, rf, TOF*u.day)

    # from astropy import time
    # from astropy import units as u

    # import numpy as np
    # from poliastro.twobody.propagation import propagate, cowell
    # from poliastro.core.perturbations import J2_perturbation
    # from poliastro.core.propagation import func_twobody


    # def f(t0, state, k):
    #     du_kep = func_twobody(t0, state, k)
    #     ax, ay, az = J2_perturbation(
    #         t0, state, k, J2=Sun.J2.value, R=Sun.R.to(u.km).value
    #     )
    #     du_ad = np.array([0, 0, 0, ax, ay, az])

    #     return du_kep + du_ad

    # times = np.linspace(0, TOF*u.day, 500)

    # positions = propagate(
    #     TFC_orb,
    #     time.TimeDelta(times),
    #     method=cowell,
    #     rtol=1e-11,
    #     f=f
    # )

    # norm([positions[-1].x.value, positions[-1].y.value, positions[-1].z.value]*u.km - rf)

    return norm(pk_error.value), norm(TFC_error.value), ratio.value