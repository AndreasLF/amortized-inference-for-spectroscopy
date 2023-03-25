import numpy as np
from matplotlib import pyplot as plt
import pandas as pd
import torch
# import data
from datetime import date 
import os
import sys

# Get working directory, parent directoy, data and results directory
cwd = os.getcwd()
curr_script_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(curr_script_dir)
results_dir = os.path.join(parent_dir, 'results')
data_dir = os.path.join(parent_dir, 'data')
# Add data simulator to path
sys.path.append(curr_script_dir + "/data_simulator")
# Import data simulator
from SERSGenerator import SERSGenerator, pseudo_voigt


def generate_SERS_dataset(data_points):
    # define the generator without a seed
    # gen = SERSGenerator((20,20), 500, seed=None, eta=[0,0])

    # dataframe to store the data
    df = pd.DataFrame(columns=['X', 'y'])
    X_col = []
    y_col = []

    for i in range(data_points):   
        # Randmon float between 0 and 500
        # gen.c = np.random.uniform(0,500, size=(1,1))
        gen = SERSGenerator((20,20), 500, seed=None, eta=[0,0])

        gen.generate(1,1,0.1,1, plot=False, background='none')
        X = np.array(gen.Vp[0])
        y = gen.c

        X_col.append(X)
        y_col.append(y)
    
    
    # Make X into dataframe
    df = pd.DataFrame(np.array(X_col))
    # add y as column
    df['y'] = np.array(y_col)

    return df

def SERS_generator_function(single_spectrum = True, Nw = 500, mapsize = (20,20), num_peaks=1, num_hotspots=1, eta = [0,0], gamma = None, seed = None):
    """ Generator function for SERS data. Generates a batch of SERS spectra.
    Args:
        batch_size (int): Number of spectra in the batch    
        single_spectrum (bool): If True, only the spectrum with the highest intensity is returned
        Nw (int): Number of wavenumbers
        mapsize (tuple): Size of the map
        num_peaks (int): Number of peaks in the spectrum
        num_hotspots (int): Number of hotspots in the spectrum
        eta (list): List of two floats, the first is the lower bound and the second is the upper bound of the eta parameter
        gamma (list): List of two floats, the first is the lower bound and the second is the upper bound of the gamma parameter
        seed (int): Seed for the random number generator

    Returns:
        X (np.array): Array of shape (batch_size, Nw) containing the SERS spectra
        Vp (np.array): Array of shape (batch_size, Nw) containing the pure voigt profiles of the peaks
        A (np.array): Array of shape (batch_size, Nw) containing the relative amplitudes of the peaks
        c (np.array): Array of shape (batch_size, 1) containing the concentrations
        num_hotspots (int): Number of hotspots in the spectrum
        num_peaks (int): Number of peaks in the spectrum   

    """

    while True: 
        batch = []
        generator_params_batch = []

        if num_peaks == None:
            num_peaks = np.random.randint(1,5)
        
        if num_hotspots == None:
            num_hotspots = np.random.randint(1,5)
        
        gen = SERSGenerator(mapsize, Nw, seed=seed, c=None, eta=eta, gamma=None)

        X = gen.generate(num_hotspots, num_peaks, 0.1,1, plot=False, background='none')

        if single_spectrum:
            max_row = np.argmax(np.sum(X, axis=1))
            X = X[max_row,:]

        # Get the relative amplitudes of the peaks
        A = gen.alpha
        # Get pure voigts of the peaks
        Vp = gen.Vp
        
        generator_params = [Vp, A, gen.c, num_hotspots, num_peaks]

        X = np.array(X).astype(np.float32)
        Vp = np.array(Vp).astype(np.float32)
        Vp = Vp.flatten()

        yield X, generator_params


if __name__ == "__main__":
    # i = 0
    # for x, y in SERS_generator_function(single_spectrum = True, num_peaks=1, num_hotspots=1):
    #     print(x.shape)
    #     i += 1
    #     if i == 10:
    #         break

    # get date
    today = date.today()
    n = 1000
    split = 0.8
    n_train = int(n*split)
    n_test = int(n*(1-split))

    df_train = generate_SERS_dataset(n_train)
    df_test = generate_SERS_dataset(n_test)

    df_train.to_csv(f"{data_dir}/SERS_data/{n}_SERS_train_data_{today}.csv", index=False)
    df_test.to_csv(f"{data_dir}/SERS_data/{n}_SERS_test_data_{today}.csv", index=False)



