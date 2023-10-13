import hipopy.hipopy as hp # <--- Package for reading in the hipo files
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rc
import ROOT
import timeit
import statistics
import tensorflow as tf
from tensorflow import keras as keras
from tensorflow.keras.preprocessing.sequence import pad_sequences
import random # making random colors
import os

import sys #NOTE: ADDED
import glob #NOTE: ADDED

def load_file_list(file_list,delimiter=',',dtype=np.float):
    arr = []
    for path in file_list:
        a = np.loadtxt(path,delimiter=",", dtype=dtype)
        if len(arr)==0: arr = a
        else: arr = np.concatenate(arr,a)
    
    return arr


def main(
        xyz_train_file_list,
        p_theta_phi_train_file_list,
        xyz_val_file_list,
        p_theta_phi_val_file_list,
        xyz_test_file_list,
        p_theta_phi_test_file_list
        ):

    # Load training data
    xyz_training = load_file_list(xyz_train_file_list)
    p_theta_phi_training = load_file_list(p_theta_phi_train_file_list)

    # Load validation data
    xyz_validation = load_file_list(xyz_val_file_list)
    p_theta_phi_validation= load_file_list(p_theta_phi_val_file_list)

    # Load test data
    xyz_test = load_file_list(xyz_test_file_list)
    p_theta_phi_test = load_file_list(p_theta_phi_test_file_list)

    #TODO: ------------------------------ INSERT JOSEPH'S CODE HERE.  MAKE SURE PLOTS GET SAVED THOUGH. ------------------------------#

    print(len(xyz_training), len(xyz_training[0]), len(p_theta_phi_training), len(p_theta_phi_training[0]))

    model = keras.Sequential()

    input_dim = len(xyz_training[0])                       # input dimension depending on the amount of tracking data pulled from each event in REC::Traj
    output_dim = len(p_theta_phi_training[0])              # p, theta, phi, vx, vy, and vz

    model.add(keras.layers.Dense(units=64, activation='relu', input_shape=(input_dim,))) 
    model.add(keras.layers.Dense(units=64, activation='relu')) 
    model.add(keras.layers.Dense(units=64, activation='relu'))
    model.add(keras.layers.Dense(units=64, activation='relu')) 
    model.add(keras.layers.Dense(units=64, activation='relu'))
    model.add(keras.layers.Dense(units=output_dim, activation='linear')) # linear activation?

    model.compile(loss='mean_absolute_error', optimizer='adam') # 'mean_squared_error' if the data isn't normalized

    # fitting the neural network
    model.fit(xyz_training, np.array(p_theta_phi_training), batch_size=64, epochs=10, validation_data=(xyz_validation, np.array(p_theta_phi_validation)))

    # predicting
    predictions_normalized = model.predict(xyz_test)
    print(predictions_normalized)

    p_predictions_normalized = []
    theta_predictions_normalized = []
    phi_predictions_normalized = []

    vx_predictions_normalized = []
    vy_predictions_normalized = []
    vz_predictions_normalized = []

    for _,v in enumerate(predictions_normalized):
        p_predictions_normalized.append(v[0])
        theta_predictions_normalized.append(v[1])
        phi_predictions_normalized.append(v[2])
        vx_predictions_normalized.append(v[3])
        vy_predictions_normalized.append(v[4])
        vz_predictions_normalized.append(v[5])

    # Create a grid of subplots
    fig1, axs1 = plt.subplots(2, 3, figsize=(25, 16))

    # Adjust spacing
    fig1.subplots_adjust(hspace=0.5)
    fig1.subplots_adjust(wspace=0.5)

    axs1[0,0].hist(p_predictions_normalized, bins=50, color="maroon")
    axs1[0,0].set_xlabel("momentum, p (GeV)")
    axs1[0,0].set_ylabel("Count")

    axs1[0,1].hist(theta_predictions_normalized, bins=50, color="goldenrod")
    axs1[0,1].set_xlabel("azimuthal angle (degrees)")
    axs1[0,1].set_ylabel("Count")

    axs1[0,1].set_title("p, theta, and phi normalized predictions\n\n", loc='center', fontsize=24)

    axs1[0,2].hist(phi_predictions_normalized, bins=50, color="peru")
    axs1[0,2].set_xlabel("polar angle (degrees)")
    axs1[0,2].set_ylabel("Count")


    axs1[1,0].hist(vx_predictions_normalized, bins=50, color="maroon")
    axs1[1,0].set_xlabel("vx (cm)")
    axs1[1,0].set_ylabel("Count")

    axs1[1,1].hist(vy_predictions_normalized, bins=50, color="goldenrod")
    axs1[1,1].set_xlabel("vy (cm)")
    axs1[1,1].set_ylabel("Count")

    axs1[1,1].set_title("vertex normalized predictions\n\n", loc='center', fontsize=24)

    axs1[1,2].hist(vz_predictions_normalized, bins=50, color="peru")
    axs1[1,2].set_xlabel("vz (cm)")
    axs1[1,2].set_ylabel("Count")

    # plt.show()
    fig1.savefig('fig1.pdf')

    # un-normalize data

    # Un-normalize p values to the range [0, 2]
    p_predictions = [p * 2 for p in p_predictions_normalized]

    # Un-normalize theta values to the range [0, 180]
    theta_predictions = [theta * 180 for theta in theta_predictions_normalized]

    # Un-normalize phi values to the range [-180, 180]
    phi_predictions = [phi * 360 - 180 for phi in phi_predictions_normalized]

    # Un-normalize vx, vy, vz
    vx_predictions = [filler * vx_max for filler in vx_predictions_normalized]

    vy_predictions = [filler * vy_max for filler in vy_predictions_normalized]

    vz_predictions = [filler * vz_max for filler in vz_predictions_normalized]

    # Create a grid of subplots
    fig1, axs1 = plt.subplots(2, 3, figsize=(25, 16))

    # Adjust spacing
    fig1.subplots_adjust(hspace=0.5)
    fig1.subplots_adjust(wspace=0.5)

    axs1[0,0].hist(p_predictions, bins=50, color="maroon")
    axs1[0,0].set_xlabel("momentum, p (GeV)")
    axs1[0,0].set_ylabel("Count")

    axs1[0,1].hist(theta_predictions, bins=50, color="goldenrod")
    axs1[0,1].set_xlabel("azimuthal angle (degrees)")
    axs1[0,1].set_ylabel("Count")

    axs1[0,1].set_title("p, theta, and phi predictions\n\n", loc='center', fontsize=24)

    axs1[0,2].hist(phi_predictions, bins=50, color="peru")
    axs1[0,2].set_xlabel("polar angle (degrees)")
    axs1[0,2].set_ylabel("Count")


    axs1[1,0].hist(vx_predictions, bins=50, color="maroon")
    axs1[1,0].set_xlabel("vx (cm)")
    axs1[1,0].set_ylabel("Count")

    axs1[1,1].hist(vy_predictions, bins=50, color="goldenrod")
    axs1[1,1].set_xlabel("vy (cm)")
    axs1[1,1].set_ylabel("Count")

    axs1[1,1].set_title("vertex predictions\n\n", loc='center', fontsize=24)

    axs1[1,2].hist(vz_predictions, bins=50, color="peru")
    axs1[1,2].set_xlabel("vz (cm)")
    axs1[1,2].set_ylabel("Count")

    # plt.show()
    fig1.savefig('fig2.pdf')

    len(xyz_test)

    print("p vector lengths:", len(p_true), len(p_predictions), (len(p_true) * len(p_predictions)))

    # matching with cutoffs |theta_mc-theta_rec|<2deg and |phi_mc-phi_rec|<6deg, minimize quadrature sum error
    # change matching formula to also include momentum? |p_true - p_predictions|<0.05 GeV

    start_time = timeit.default_timer()

    p_predictions_changing = p_predictions[:]
    theta_predictions_changing = theta_predictions[:]
    phi_predictions_changing = phi_predictions[:]

    matching_indices = np.zeros((2, len(p_true)), dtype=int)
    unmatched_indices = np.array([], dtype=int)
    num_matches = 0

    # going through all the "true" pions
    for i,_ in enumerate(p_true):
        
        min_quad_error = 1000 # arbitrary large number
        for j,_ in enumerate(p_predictions): # for each true pion, look through every predicted pion
            
            theta_error = theta_true[i] - theta_predictions_changing[j]
            phi_error = phi_true[i] - phi_predictions_changing[j]
            p_error = p_true[i] - p_predictions_changing[j] 
            
            if (abs(theta_error) < 2 or abs(theta_error) > 178) & (abs(phi_error) < 6 or abs(phi_error) > 354): # does this true pion have a similar predicted pion, based on their angles?
                min_quad_error_temp = np.sqrt((theta_error) ** 2 + (phi_error) ** 2) + 10 * p_error # in case there are multiple hits, we can minimize based off a quadrature sum
                if (min_quad_error_temp < min_quad_error):
                    min_quad_error = min_quad_error_temp
                    predictions_index = j # saving the matched index of the "best" predicted pion
                    
        if (min_quad_error < 1000): # if there were any matches for this true pion, essentially
            matching_indices[0][num_matches] = i
            matching_indices[1][num_matches] = predictions_index # saving the matching indices of the true and predicted pions
            
            p_predictions_changing[predictions_index] = 100
            theta_predictions_changing[predictions_index] = 100
            phi_predictions_changing[predictions_index] = 100  # this is my way of avoiding duplicates - obviously angles of 3000 degrees will not be put as a match to other pions going forward
            
            num_matches += 1
        else:
            unmatched_indices = np.append(unmatched_indices, [i])
            
    matching_indices = matching_indices[:, :num_matches]

    print("time computing", timeit.default_timer() - start_time)
    print(matching_indices)
    print(len(p_true) * len(p_predictions))

    # finding the difference between the true and predicted data
    p_difference = []
    theta_difference = []
    phi_difference = []

    for i,_ in enumerate(matching_indices[1]):
        p_difference.append(p_true[matching_indices[0][i]] - p_predictions[matching_indices[1][i]])
        theta_difference.append(theta_true[matching_indices[0][i]] - theta_predictions[matching_indices[1][i]])
        phi_difference.append(phi_true[matching_indices[0][i]] - phi_predictions[matching_indices[1][i]])

    p_difference = np.array(p_difference)
    theta_difference = np.array(theta_difference)
    phi_difference = np.array(phi_difference)

    # Create a grid of subplots
    fig1, axs1 = plt.subplots(1, 3, figsize=(25, 12))

    # Adjust spacing
    fig1.subplots_adjust(hspace=0.5)
    fig1.subplots_adjust(wspace=0.5)

    # predicted data resolution histograms

    axs1[0].hist(theta_difference, bins=100, color="fuchsia")
    axs1[0].set_ylabel("Count")
    axs1[0].set_xlabel(r"$\theta$ difference (degrees)")

    axs1[1].hist(phi_difference, bins=3000, color="mediumblue")
    axs1[1].set_ylabel("Count")
    axs1[1].set_xlabel(r"$\phi$ difference (degrees)")
    axs1[1].set_xlim(-6, 6)

    axs1[2].hist(p_difference, bins=100, color="darkseagreen")
    axs1[2].set_ylabel("Count")
    axs1[2].set_xlabel(r"p difference (GeV)")


    plt.suptitle("Angle and momentum resolution", fontsize=24)

    # plt.show()
    fig1.savefig('fig3.pdf')

    return

# ------------------------------ END MAIN DEFINITION ------------------------------ #

#------------------------------ MAIN ------------------------------#
if __name__=="__main__":
    if len(sys.argv)<=7:
        print(
        "Usage: ",os.path.abspath(sys.argv[0]),
        " 'xyz_train_regex' 'ptp_train_regex' 'xyz_val_regex' 'ptp_val_regex' 'xyz_test_regex' 'ptp_test_regex'"
        )
        sys.exit(0)

    xyz_train_file_list = [os.path.abspath(el) for el in glob.glob(sys.argv[1])]
    ptp_train_file_list = [os.path.abspath(el) for el in glob.glob(sys.argv[2])]
    xyz_val_file_list   = [os.path.abspath(el) for el in glob.glob(sys.argv[3])]
    ptp_val_file_list   = [os.path.abspath(el) for el in glob.glob(sys.argv[4])]
    xyz_test_file_list  = [os.path.abspath(el) for el in glob.glob(sys.argv[5])]
    ptp_test_file_list  = [os.path.abspath(el) for el in glob.glob(sys.argv[6])]
    main(
        xyz_train_file_list,
        ptp_train_file_list,
        xyz_val_file_list,
        ptp_val_file_list,
        xyz_test_file_list,
        ptp_test_file_list
        )
