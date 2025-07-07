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
    gamma = 5/3
    eps = Pth/(gamma-1)
    return eps

def kinetic_energy(dictionary, time):
    emag = magnetic_energy(dictionary, time)
    eps = internal_energy(dictionary, time)
    E = dictionary[time]['hydro']['data'][0, 3:-3, 3:-3, 1]
    ekin = E - emag - eps
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

def velocity_components(dictionary, time):
    v_x = dictionary[time]['hydro']['data'][0, 3:-3, 3:-3, 2]
    v_y = dictionary[time]['hydro']['data'][0, 3:-3, 3:-3, 3]
    v_z = dictionary[time]['hydro']['data'][0, 3:-3, 3:-3, 4]
    return v_x, v_y, v_z

def alfven_velocity(dictionary, time):
    B = np.sqrt(magnetic_field(dictionary, time))
    sqrrho = np.sqrt(dictionary[time]['hydro']['data'][0, 3:-3, 3:-3, 0])
    v_A = B / sqrrho
    return v_A

def fluid_velocity(dictionary, time):
    rho = dictionary[time]['hydro']['data'][0, 3:-3, 3:-3, 0]
    ekin = kinetic_energy(dictionary, time)
    v_vec = np.sqrt(2 * ekin / rho)
    return v_vec

def mach_number(dictionary, time):
    v = fluid_velocity(dictionary, time)
    c_s = dictionary[time]['thd']['data'][0, 3:-3, 3:-3, 1]
    mach = v / c_s
    return mach

def alfven_number(dictionary, time):
    v = fluid_velocity(dictionary, time)
    v_alfven = alfven_velocity(dictionary, time)
    alf_number = v / v_alfven
    return alf_number

def cantidad_promedio(data):
    data = np.array(data)
    if data.ndim == 3:
        n_time, Nx, Ny = data.shape
        dx, dy = 1 / Nx, 1 / Ny
        E = np.sum(data, axis=(1, 2)) * dx * dy
        return E
    elif data.ndim == 2:
        Nx, Ny = data.shape
        dx, dy = 1 / Nx, 1 / Ny
        E = np.sum(data) * dx * dy
        return E

def kinetic_helicity(dictionary, time):
    vx, vy, _ = velocity_components(dictionary, time)
    Bx, By, _ = magnetic_field_components(dictionary, time)
    Nx, Ny = vx.shape
    dx, dy = 1 / Nx, 1 / Ny
    dvy_dx = np.gradient(vy, dx, axis=0)
    dvx_dy = np.gradient(vx, dy, axis=1)
    curl_v_z = dvy_dx - dvx_dy
    Hk = (vx * curl_v_z + vy * curl_v_z).sum() * dx * dy
    return Hk

def cross_helicity(dictionary, time):
    vx, vy, _ = velocity_components(dictionary, time)
    Bx, By, _ = magnetic_field_components(dictionary, time)
    Nx, Ny = vx.shape
    dx, dy = 1 / Nx, 1 / Ny
    Hc = (vx * Bx + vy * By).sum() * dx * dy
    return Hc

def magnetic_helicity(dictionary, time):
    vx, vy, _ = velocity_components(dictionary, time)
    Bx, By, _ = magnetic_field_components(dictionary, time)
    Nx, Ny = vx.shape
    dx, dy = 1 / Nx, 1 / Ny
    Bx_k = np.fft.fft2(Bx)
    By_k = np.fft.fft2(By)
    kx = np.fft.fftfreq(Nx, d=dx) * 2 * np.pi
    ky = np.fft.fftfreq(Ny, d=dy) * 2 * np.pi
    KX, KY = np.meshgrid(kx, ky, indexing='ij')
    K2 = KX**2 + KY**2
    K2[0, 0] = 1
    Az_k = 1j * (KX * By_k - KY * Bx_k) / K2
    Az_k[0, 0] = 0
    Az = np.fft.ifft2(Az_k).real
    dAz_dy = np.gradient(Az, dy, axis=1)
    dAz_dx = np.gradient(Az, dx, axis=0)
    Hm = (Az * (dAz_dy - (-dAz_dx))).sum() * dx * dy
    return Hm
