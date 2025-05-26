import os
import pandas as pd
from tqdm import tqdm

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

from numpy import diff, concatenate, array, ones, empty, stack, sqrt, arange, zeros, pi, real, append
import numpy as np
import h5py

from scipy.ndimage import convolve
from scipy.fft import fftn, ifftn, fftshift, ifftshift, fftfreq


# =============================================================================
# Abrir los archivos h5 y abrir una capeta con todos los archivos h5 y guardarlos en un diccionario
# =============================================================================
def open_hdf5_file(filename):
    def read_group(group):
        data = {}
        for key, item in group.items():
            if isinstance(item, h5py.Group):  # If it's a group, recurse
                data[key] = read_group(item)
            elif isinstance(item, h5py.Dataset):  # If it's a dataset, read it
                data[key] = item[()]
        return data

    with h5py.File(filename, "r") as f:
        data = read_group(f)  # Start reading from the root group
    return data


def open_all_hdf5_file(folder_path):
    all_files = os.listdir(folder_path)
    dictionary = {}
    for i in range(len(all_files)):
        file = all_files[i]
        if file.endswith('.h5'):
            dictionary[i] = open_hdf5_file(os.path.join(folder_path, file))
        else:
            None
    return dictionary

def open_dat_files(folder_path: str, flag: str):
    """
    Loads all .dat files in the specified folder using corresponding header (.txt) files.
    
    For each .dat file in the folder, the function looks for a .txt file with the same base name 
    (which contains the header names). The .dat file is read into a DataFrame using these headers.
    
    Parameters:
        folder_path (str): Path to the folder containing the .dat and .txt files.
        flag (str): A flag string (can be used to modify behavior or specify an output directory for plots).
    
    Returns:
        list: A list of pandas DataFrames loaded from the .dat files.
    """
    dataframes = []
    
    # Iterate over all files in the folder
    for file in tqdm(os.listdir(folder_path)):
        # Process only .dat files
        if file.endswith('.dat'):
            dat_file = file
            header_file = file.replace('.dat', '.txt')
            header_path = os.path.join(folder_path, header_file)
            
            # Check if the corresponding header (.txt) file exists
            if not os.path.exists(header_path):
                print(f"Header file {header_file} not found for {dat_file} skipping.")
                continue
            
            # Read the header file to obtain column names
            try:
                with open(header_path, 'r') as hf:
                    header_line = hf.readline().strip()
                    headers = header_line.split()  # Assumes headers are space-separated
            except Exception as e:
                print(f"Error reading header file {header_file}: {e}")
                continue
            
            # Read the .dat file using the header names
            dat_path = os.path.join(folder_path, dat_file)
            try:
                df = pd.read_csv(dat_path, delimiter=r'\s+', names=headers, header=None)
                dataframes.append(df)
                print(f"Archivo cargado: {dat_file} con encabezados: {headers}")
            except Exception as e:
                print(f"Error al cargar {dat_file}: {e}")
    
    print(f"Se cargaron {len(dataframes)} archivos .dat")
    return dataframes

# Example usage:
# folder_path = r'C:\Users\carlo\Desktop\4 FISICA\TFM\Resultados TFM\OT_test_hlld_60seconds\log'
# flag = r'C:\Users\carlo\Desktop\4 FISICA\TFM\Resultados TFM\Results\Plots 1D\hlld_60s'
# dataframes = load_dat_files(folder_path, flag)

# libro_erg, libro_grw, libro_mag, libro_mom, libro_neu, libro_rho, libro_tim, libro_vel = dataframes[0], dataframes[1], dataframes[2], dataframes[3], dataframes[4], dataframes[5], dataframes[6], dataframes[7]

# =============================================================================
# Espectros from Carlos Rabano
# =============================================================================
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

# Animation and plot 2D
def simple_colormap_2D(data, title=None, save_path=None):
    # Si data es 3D (por ejemplo, múltiples frames), se toma el primero
    if data.ndim == 3:
        data_to_plot = data[0]
    else:
        data_to_plot = data

    fig, ax = plt.subplots(figsize=(10, 10))

    # Mostrar la imagen
    # cax = ax.imshow(data_to_plot, cmap='turbo', vmin=np.min(data_to_plot), vmax=2)
    cax = ax.imshow(data_to_plot, cmap='turbo', vmin=np.min(data_to_plot), vmax=np.max(data_to_plot))
    
    # Ticks del eje
    ax.tick_params(
        which='both',
        bottom=True, top=True,
        left=True, right=True,
        labelbottom=False, labelleft=True,
        labeltop=True
    )
    
    # Añadir colorbar con ajustes de tamaño
    cbar = plt.colorbar(cax, ax=ax, shrink=0.8, fraction=0.05, pad=0.1)

    # Añadir título si se proporciona
    if title is not None:
        ax.set_title(title, pad=15)

    # Guardar o mostrar
    if save_path:
        plt.savefig(save_path, bbox_inches='tight', dpi=300)
    plt.show()

# from matplotlib import FuncAnimation

def animate_2D(data, title=None, save_path=None, writer='Pillow'):
    fig1, ax1 = plt.subplots(figsize=(10,10))
    if title:
        fig1.suptitle(title, fontsize=14)
    cax1 = ax1.imshow(data[0], cmap='turbo', vmin=np.min(data), vmax=np.max(data))
    plt.colorbar(cax1, fraction=0.046, pad=0.04)
    ax1.tick_params(
        which='both',
        bottom=True, top=True,
        left=True, right=True,
        labelbottom=False, labelleft=True,
        labeltop=True
    )
    frame_text = ax1.text(0.02, 0.95, '',color='white',transform=ax1.transAxes,fontsize=12,verticalalignment='top',bbox=dict(facecolor='black', alpha=0.5)) 
    def animate(frame):
        cax1.set_array(data[frame])
        frame_text.set_text(f'Frame: {frame + 1}/{len(data)}')
        return cax1, frame_text

    ani1 = FuncAnimation(fig1, animate,frames=len(data),interval=10,blit=True) 
    if save_path:
        ani1.save(save_path, writer)
    plt.show()



# Physical functions
def magnetic_field(dictionary, time):
    B = np.sqrt(dictionary[time]['mag_vol']['data'][0, 3:-3, 3:-3, 0]**2 + dictionary[time]['mag_vol']['data'][0, 3:-3, 3:-3, 1]**2)
    return B

def alfven_velocity(dictionary, time):
    # data = dictionary[time]['mag_vol']['data'][0, 3:-3, 3:-3, 1]

    B = np.sqrt(magnetic_field(dictionary, time))
    sqrrho = np.sqrt(dictionary[time]['hydro']['data'][0, 3:-3, 3:-3, 0])
    v_A = B / sqrrho
    return v_A

def mach_numer(dict, libro, time):
    data = dict[time]['mag_vol']['data'][0, 3:-3, 3:-3, 1]
    c_s = dict[time]['thd']['data'][0, 3:-3, 3:-3, 1]
    v_A = alfven_velocity(dict, time)
    v = np.sqrt(libro[-1]['vv_xx_time,'].to_numpy() + libro[-1]['vv_yy_time,'].to_numpy())
    v = v[time]
    Mach = v / c_s
    return Mach