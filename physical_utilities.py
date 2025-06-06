import os
import numpy as np

def magnetic_field(dictionary, time):
    B = np.sqrt(dictionary[time]['mag_vol']['data'][0, 3:-3, 3:-3, 0]**2 + dictionary[time]['mag_vol']['data'][0, 3:-3, 3:-3, 1]**2)
    return B

def magnetic_field_components(dictionary, time):
    Bx = dictionary[time]['mag_vol']['data'][0, 3:-3, 3:-3, 0]
    By = dictionary[time]['mag_vol']['data'][0, 3:-3, 3:-3, 1]
    Bz = dictionary[time]['mag_vol']['data'][0, 3:-3, 3:-3, 2]
    return Bx, By, Bz

def magnetic_energy(dictionary, time):
    B = np.sqrt(dictionary[time]['mag_vol']['data'][0, 3:-3, 3:-3, 0]**2 + dictionary[time]['mag_vol']['data'][0, 3:-3, 3:-3, 1]**2)
    return B**2/2

def internal_energy(dictionary, time):
    Pth = dictionary[time]['thd']['data'][0, 3:-3, 3:-3, 0]
    gamma = 5/3 # polytrope non-relativistic. If relativistic, use gamma = 4/3
    eps = Pth/(gamma-1)
    return eps

def kinetic_energy(dictionary, time):
    emag = magnetic_energy(dictionary, time)
    eps = internal_energy(dictionary, time)
    E = dictionary[time]['hydro']['data'][0, 3:-3, 3:-3, 1]
    ekin = E-emag-eps
    return ekin

def total_energy(dictionary, time):
    E = dictionary[time]['hydro']['data'][0, 3:-3, 3:-3, 1]
    return E

def thermal_pressure(dictionary, time):
    p_th = dictionary[time]['thd']['data'][0, 3:-3, 3:-3, 0]
    return p_th

def fluid_density(dictionary, time):
    rho = dictionary[time]['hydro']['data'][0, 3:-3, 3:-3, 0]
    return rho

def total_pressure_components(dictionary, time):
    p_x = dictionary[time]['hydro']['data'][0, 3:-3, 3:-3, 2]
    p_y = dictionary[time]['hydro']['data'][0, 3:-3, 3:-3, 3]
    p_z = dictionary[time]['hydro']['data'][0, 3:-3, 3:-3, 4]
    return p_x, p_y, p_z

def alfven_velocity(dictionary, time):
    B = np.sqrt(magnetic_field(dictionary, time))
    sqrrho = np.sqrt(dictionary[time]['hydro']['data'][0, 3:-3, 3:-3, 0])
    v_A = B / sqrrho
    return v_A

def fluid_velocity(dictionary, time):
    rho = dictionary[time]['hydro']['data'][0, 3:-3, 3:-3, 0]
    ekin = kinetic_energy(dictionary, time)
    v_vec = np.sqrt(2*ekin/rho)
    return v_vec

def mach_number(dictionary, time):
    v = fluid_velocity(dictionary, time)
    c_s = dictionary[time]['thd']['data'][0, 3:-3, 3:-3, 1]
    mach = v/c_s
    return mach

def alfven_number(dictionary, time):
    v = fluid_velocity(dictionary, time)
    v_alfven = alfven_velocity(dictionary, time)
    alf_number = v / v_alfven
    return alf_number