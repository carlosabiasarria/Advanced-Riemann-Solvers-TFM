import os
import numpy as np
from numpy import diff, concatenate, array, ones, empty, stack, sqrt, arange, zeros, pi, real, append
from scipy.ndimage import convolve
from scipy.fft import fftn, ifftn, fftshift, ifftshift, fftfreq

def grid_widths(field, dims=3):
    if field.shape[0] == dims:
        return array([field.shape[i + 1] for i in range(dims)])
    else:
        return array([field.shape[i] for i in range(dims)])

def get_spectrum(field, norm='forward', zero_frequency=False, dims=3):
    if field.shape[0] == dims:
        field_mean = field.mean(tuple(range(1, dims + 1)))
        power_spectrum = sum([abs(fftshift(fftn(field[i] - field_mean[i]))) ** 2 for i in range(dims)])
    else:
        field_mean = field.mean()
        power_spectrum = abs(fftshift(fftn(field - field_mean)))
    if (norm == 'forward') or (norm == 'ortho'):
        Normfactor = field.shape[-1]
        for i in range(2, dims + 1):
            Normfactor = Normfactor * field.shape[-i]
        if norm == 'forward':
            Normfactor = Normfactor ** 2
        power_spectrum = power_spectrum / Normfactor
    elif norm == 'backward':
        pass
    else:
        raise ValueError(f'Invalid norm value {norm}, should be "forward", "ortho" or "backward".')
    k = []
    dk = []
    for i in range(dims):
        rs = [1]*dims
        rs[i] = -1
        rs = tuple(rs)
        dki = 2 * pi / field.shape[-(i + 1)]
        dk.append(dki)
        ki = dki * field.shape[-(i + 1)] * fftshift(fftfreq(field.shape[-(i + 1)])).reshape(rs)
        k.append(ki)
    k = sqrt(sum([ki**2 for ki in k]))
    dk = min(dk)
    k_bins = arange(dk, k.max() + dk, dk)
    energy = zeros(len(k_bins) - 1)
    for i in range(len(k_bins) - 1):
        mask = (k >= k_bins[i]) & (k < k_bins[i + 1])
        energy[i] = power_spectrum[mask].sum()
    k = k_bins[:-1] + dk / 2
    if zero_frequency:
        k = append(0, k)
        if field.shape[0] == 3:
            energy0 = (field_mean**2).sum()
        else:
            energy0 = field_mean**2
        energy = append(energy0, energy)
    return k, energy

def radial_spectrum(field, L=1.0, dims=2, coords=None):
    f = np.asarray(field)
    if dims not in (1,2,3):
        raise ValueError("dims must be 1, 2 or 3")
    if f.ndim != dims:
        raise ValueError(f"field.ndim={f.ndim} does not match dims={dims}")
    if np.isscalar(L):
        L = [L]*dims
    L = list(L)
    if len(L) != dims:
        raise ValueError(f"Length of L ({len(L)}) must equal dims ({dims})")
    if coords is not None and dims == 1:
        x = np.asarray(coords[0], float)
        if x.ndim != 1 or x.size != f.size:
            raise ValueError("coords[0] must be a 1D array matching field size")
        dx = np.diff(x, prepend=x[0])
        N = f.size
        nbins = N//2
        k_vals = np.arange(1, nbins+1)
        E_k = np.zeros(nbins)
        counts = np.ones(nbins, int)
        for idx, k in enumerate(k_vals):
            Fk = np.sum(f * np.exp(-2j*np.pi * k * x) * dx)
            E_k[idx] = np.abs(Fk)**2
        return k_vals, E_k
    Ns = f.shape
    dels = [L[i]/Ns[i] for i in range(dims)]
    F = fftn(f)
    P = np.abs(fftshift(F))**2
    ks = []
    for i in range(dims):
        ki = fftshift(fftfreq(Ns[i], d=dels[i]))
        shape = [1]*dims
        shape[i] = Ns[i]
        ks.append(ki.reshape(shape))
    k_mag = np.sqrt(sum(ki**2 for ki in ks))
    k_int = np.rint(k_mag).astype(int)
    kmax = Ns[0]//2
    k_vals = np.arange(1, kmax+1)
    E_k = np.zeros_like(k_vals, float)
    counts = np.zeros_like(k_vals, int)
    flatP = P.ravel()
    flatk = k_int.ravel()
    for idx, k in enumerate(k_vals):
        mask = (flatk == k)
        counts[idx] = mask.sum()
        if counts[idx] > 0:
            E_k[idx] = flatP[mask].mean()
    E_k[-1] = 0.0
    return k_vals, E_k
