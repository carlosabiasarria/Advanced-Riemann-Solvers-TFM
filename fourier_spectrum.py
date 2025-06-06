import os
import numpy as np
from numpy import diff, concatenate, array, ones, empty, stack, sqrt, arange, zeros, pi, real, append
from scipy.ndimage import convolve
from scipy.fft import fftn, ifftn, fftshift, ifftshift, fftfreq


def grid_widths(field, dims=3):
    """
    Calculate the grid widths for a given field.

    Parameters:
    field (numpy.ndarray): The input field.
    dims (int): The number of dimensions of the field.

    Returns:
    numpy.ndarray: The grid widths.
    """
    if field.shape[0] == dims:
        return array([field.shape[i + 1] for i in range(dims)])
    else:
        return array([field.shape[i] for i in range(dims)])


def get_spectrum(field, norm='forward', zero_frequency=False, dims=3):
    # Remove the mean of the field (components) to eliminate the zero-frequency contribution
    # and reduce numerical errors, as it is usually the largest one.
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
            # Normalization in Fourier space.
            Normfactor = Normfactor ** 2
        power_spectrum = power_spectrum / Normfactor
    elif norm == 'backward':
        # Normalization in physical space
        pass
    else:
        raise ValueError(f'Invalid norm value {norm}, should be "forward", "ortho" or "backward".')
    k = []
    dk = []
    for i in range(dims):
        rs = [1]*dims
        rs[i] = -1
        rs = tuple(rs)
        dki = 2 * pi / field.shape[-(i + 1)]  # The factor of 2 is because the FFT is normalized by the number of points.
        dk.append(dki)
        ki = dki * field.shape[-(i + 1)] * fftshift(fftfreq(field.shape[-(i + 1)])).reshape(rs)
        k.append(ki)
    k = sqrt(sum([ki**2 for ki in k]))
    dk = min(dk)
    # The zero-frequency contribution was removed.
    k_bins = arange(dk, k.max() + dk, dk)
    energy = zeros(len(k_bins) - 1)
    for i in range(len(k_bins) - 1):
        mask = (k >= k_bins[i]) & (k < k_bins[i + 1])
        energy[i] = power_spectrum[mask].sum()
    k = k_bins[:-1] + dk / 2
    if zero_frequency:
        # Add the zero-frequency contribution.
        k = append(0, k)
        if field.shape[0] == 3:
            energy0 = (field_mean**2).sum()
        else:
            energy0 = field_mean**2
        energy = append(energy0, energy)
    return k, energy


def radial_spectrum(field, L=1.0, dims=2, coords=None):
    """
    Compute the 1D radial spectrum E(k) vs. integer k shells (cycles per unit length)
    for a real field in 1D, 2D, or 3D on a possibly anisotropic uniform grid.
    Optionally supports direct 1D transforms on non-uniform grids via coords.

    The Nyquist shell is included and will naturally reflect the (near-)zero power
    if no energy resides at k = N/2.

    Parameters
    ----------
    field : ndarray
        Real input array of shape (N,) if dims=1, (Nx,Ny) if dims=2, (Nx,Ny,Nz) if dims=3.
    L     : float or sequence of floats
        Physical box size along each axis. If scalar, assumes equal lengths for all dims.
    dims  : int
        Number of physical dimensions of `field` (1, 2 or 3).
    coords : None or sequence of 1D arrays
        If provided (length=dims), `coords[i]` gives the non-uniform coordinate positions along axis i.
        Only 1D direct transforms are supported; for dims>1 mixed uniform/non-uniform, an FFT
        with mean spacing is used.

    Returns
    -------
    k_vals : ndarray
        Integer wavenumbers from 1 to floor(N/2) (cycles per unit length).
    E_k    : ndarray
        Mean power |F|^2 in each k shell, including Nyquist if present.
    counts : ndarray
        Number of modes per shell.
    """
    f = np.asarray(field)
    if dims not in (1,2,3):
        raise ValueError("dims must be 1, 2 or 3")
    if f.ndim != dims:
        raise ValueError(f"field.ndim={f.ndim} does not match dims={dims}")

    # Handle L as scalar or sequence
    if np.isscalar(L):
        L = [L]*dims
    L = list(L)
    if len(L) != dims:
        raise ValueError(f"Length of L ({len(L)}) must equal dims ({dims})")

    # 1D direct transform for non-uniform coords
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
        return k_vals, E_k#, counts

    # Uniform (or approximated) grid for dims >= 1
    Ns = f.shape
    dels = [L[i]/Ns[i] for i in range(dims)]

    # FFT & power
    F = fftn(f)
    P = np.abs(fftshift(F))**2

    # Build per-axis k_i (cycles per unit length)
    ks = []
    for i in range(dims):
        ki = fftshift(fftfreq(Ns[i], d=dels[i]))
        shape = [1]*dims
        shape[i] = Ns[i]
        ks.append(ki.reshape(shape))
    k_mag = np.sqrt(sum(ki**2 for ki in ks))

    # Radial integer binning
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

    E_k[-1] = 0.0  # Set Nyquist bin to zero if present

    return k_vals, E_k#, counts