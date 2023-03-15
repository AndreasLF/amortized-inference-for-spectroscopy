import numpy as np
from matplotlib import pyplot as plt
import pandas as pd
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

if __name__ == "__main__":
    # get date
    today = date.today()
    n = 1000

    df = generate_SERS_dataset(n)
    df.to_csv(f"{data_dir}/SERS_data/{n}_SERS_data_{today}.csv", index=False)



