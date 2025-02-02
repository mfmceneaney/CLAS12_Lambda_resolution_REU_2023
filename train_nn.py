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

def load_file_list(file_list,delimiter=',',dtype=float):
    arr = []
    for path in file_list:
        a = np.loadtxt(path,delimiter=",", dtype=dtype)
        if len(arr)==0: arr = a
        else: arr = np.concatenate((arr,a))
    
    return arr


def main(
        xyz_file_list,
        ptp_file_list,
        split,
        ):

    # Load training data
    xyz_full = load_file_list(xyz_file_list)
    ptp_full = load_file_list(ptp_file_list)

    # Get split indices
    max_len = len(xyz_full)
    cum_frac = 0.0
    split_indices = []
    for frac in split:
        cum_frac += frac
        if cum_frac>=1.0: break
        idx = int(cum_frac*max_len)
        split_indices.append(idx)

    # Split into train, test val
    xyz_training           = xyz_full[:split_indices[0]]
    p_theta_phi_training   = ptp_full[:split_indices[0]]
    xyz_validation         = xyz_full[split_indices[0]:split_indices[1]]
    p_theta_phi_validation = ptp_full[split_indices[0]:split_indices[1]]
    xyz_test               = xyz_full[split_indices[1]:]
    p_theta_phi_test       = ptp_full[split_indices[1]:]

    print("INFO: xyz, ptp full  shape = ",np.shape(xyz_training),np.shape(p_theta_phi_training))
    print("INFO: xyz, ptp train shape = ",np.shape(xyz_training),np.shape(p_theta_phi_training))
    print("INFO: xyz, ptp val   shape = ",np.shape(xyz_validation),np.shape(p_theta_phi_validation))
    print("INFO: xyz, ptp test  shape = ",np.shape(xyz_test),np.shape(p_theta_phi_test))

    # Get truth info from ptp here
    print("DEBUGGING: np.shape(ptp_full) = ",np.shape(ptp_full))
    p_true, theta_true, phi_true, vx_true, vy_true, vz_true = zip(np.swapaxes(ptp_full,0,1))
    p_true     =  np.squeeze(p_true)
    theta_true = np.squeeze(theta_true)
    phi_true   = np.squeeze(phi_true)
    vx_true    = np.squeeze(vx_true)
    vy_true    = np.squeeze(vy_true)
    vz_true    = np.squeeze(vz_true)
    print("INFO: np.shape(p_true), np.shape(theta_true), np.shape(phi_true) = ",np.shape(p_true),np.shape(theta_true),np.shape(phi_true))
    print("INFO: np.shape(vx_true), np.shape(vy_true), np.shape(vz_true) = ",np.shape(vx_true),np.shape(vy_true),np.shape(vz_true))

    # Un-normalize theta values to the range [0, 1]
    theta_true = [theta * 180 for theta in theta_true]

    # Un-normalize phi values to the range [0, 1]
    phi_true = [(phi * 360) -180 for phi in phi_true]

    # Un-normalize p values to the range [0, 1]
    p_true = [p * 2 for p in p_true]

    # Set vertex normalization values
    vx_max = 5
    vy_max = 5
    vz_max = 20

    # Un-normalize vertex values
    vx_true = vx_true * vx_max
    vy_true = vy_true * vy_max
    vz_true = vz_true * vz_max

    # Create a grid of subplots
    fig1, axs1 = plt.subplots(2, 3, figsize=(25, 16))

    # Adjust spacing
    fig1.subplots_adjust(hspace=0.5)
    fig1.subplots_adjust(wspace=0.5)

    axs1[0,0].hist(p_true, bins=50, color="maroon")
    axs1[0,0].set_xlabel("momentum, p (GeV)")
    axs1[0,0].set_ylabel("Count")

    axs1[0,1].hist(theta_true, bins=50, color="goldenrod")
    axs1[0,1].set_xlabel("polar angle (degrees)")
    axs1[0,1].set_ylabel("Count")

    axs1[0,1].set_title("p, theta, and phi normalized predictions\n\n", loc='center', fontsize=24)

    axs1[0,2].hist(phi_true, bins=50, color="peru")
    axs1[0,2].set_xlabel("azimuthal angle (degrees)")
    axs1[0,2].set_ylabel("Count")


    axs1[1,0].hist(vx_true, bins=50, color="maroon")
    axs1[1,0].set_xlabel("vx (cm)")
    axs1[1,0].set_ylabel("Count")

    axs1[1,1].hist(vy_true, bins=50, color="goldenrod")
    axs1[1,1].set_xlabel("vy (cm)")
    axs1[1,1].set_ylabel("Count")

    axs1[1,1].set_title("vertex normalized predictions\n\n", loc='center', fontsize=24)

    axs1[1,2].hist(vz_true, bins=50, color="peru")
    axs1[1,2].set_xlabel("vz (cm)")
    axs1[1,2].set_ylabel("Count")

    # plt.show()
    fig1.savefig('fig0.pdf')

    # Create a grid of subplots
    fig1, axs1 = plt.subplots(6, 6, figsize=(36, 36))

    # Adjust spacing
    fig1.subplots_adjust(hspace=0.5)
    fig1.subplots_adjust(wspace=0.5)

    ny, nx = axs1.shape
    for i in range(0,ny):
        for j in range(0,nx):
            col = i*nx + j
            print("DEBUGGING: i, j, col = ",i,j,col)
            axs1[i,j].hist(xyz_full[:,col], bins=50, color="maroon")
            axs1[i,j].set_xlabel("xyz_full[:,"+str(col)+"]")
            axs1[i,j].set_ylabel("Count")

    fig1.savefig('fig0_xyz.pdf')

   # Create a grid of subplots
    fig1, axs1 = plt.subplots(2, 3, figsize=(36, 36))

    # Adjust spacing
    fig1.subplots_adjust(hspace=0.5)
    fig1.subplots_adjust(wspace=0.5)

    ny, nx = axs1.shape
    for i in range(0,ny):
        for j in range(0,nx):
            col = i*nx + j
            print("DEBUGGING: i, j, col = ",i,j,col)
            axs1[i,j].hist(ptp_full[:,col], bins=50, color="maroon")
            axs1[i,j].set_xlabel("ptp_full[:,"+str(col)+"]")
            axs1[i,j].set_ylabel("Count")

    fig1.savefig('fig0_ptp.pdf')

    # Create a grid of subplots
    fig1, axs1 = plt.subplots(2, 3, figsize=(36, 36))

    # Adjust spacing
    fig1.subplots_adjust(hspace=0.5)
    fig1.subplots_adjust(wspace=0.5)

    ny, nx = axs1.shape
    for i in range(0,ny):
        for j in range(0,nx):
            col = i*nx + j
            print("DEBUGGING: i, j, col = ",i,j,col)
            axs1[i,j].hist(p_theta_phi_training[:,col], bins=50, color="maroon")
            axs1[i,j].set_xlabel("p_theta_phi_training[:,"+str(col)+"]")
            axs1[i,j].set_ylabel("Count")

    fig1.savefig('fig0_p_theta_phi_training.pdf')

    # Create a grid of subplots
    fig1, axs1 = plt.subplots(2, 3, figsize=(36, 36))

    # Adjust spacing
    fig1.subplots_adjust(hspace=0.5)
    fig1.subplots_adjust(wspace=0.5)

    ny, nx = axs1.shape
    for i in range(0,ny):
        for j in range(0,nx):
            col = i*nx + j
            print("DEBUGGING: i, j, col = ",i,j,col)
            axs1[i,j].hist(p_theta_phi_validation[:,col], bins=50, color="maroon")
            axs1[i,j].set_xlabel("p_theta_phi_validation[:,"+str(col)+"]")
            axs1[i,j].set_ylabel("Count")

    fig1.savefig('fig0_p_theta_phi_validation.pdf')

    # Create a grid of subplots
    fig1, axs1 = plt.subplots(2, 3, figsize=(36, 36))

    # Adjust spacing
    fig1.subplots_adjust(hspace=0.5)
    fig1.subplots_adjust(wspace=0.5)

    ny, nx = axs1.shape
    for i in range(0,ny):
        for j in range(0,nx):
            col = i*nx + j
            print("DEBUGGING: i, j, col = ",i,j,col)
            axs1[i,j].hist(p_theta_phi_test[:,col], bins=50, color="maroon")
            axs1[i,j].set_xlabel("p_theta_phi_test[:,"+str(col)+"]")
            axs1[i,j].set_ylabel("Count")

    fig1.savefig('fig0_p_theta_phi_test.pdf')

    #----- PLOT P, THETA, PHI RESOLUTION -----#                                                                                                                                 
    p_recon     = np.squeeze(xyz_full[:,-3]) * 2
    theta_recon = np.squeeze(xyz_full[:,-2]) * 180
    phi_recon   = np.squeeze(xyz_full[:,-1]) * 360 - 180

    # Create a grid of subplots                                                                                                                                                 
    fig1, axs1 = plt.subplots(1, 3, figsize=(25, 8))

    # Adjust spacing                                                                                                                                                            
    fig1.subplots_adjust(hspace=0.5)
    fig1.subplots_adjust(wspace=0.5)

    axs1[0].hist(p_true - p_recon, bins=200, color="black")
    axs1[0].set_xlabel("momentum, p resolution(GeV)")
    axs1[0].set_ylabel("Count")

    axs1[1].hist(theta_true - theta_recon, bins=200, color="blue")
    axs1[1].set_xlabel(r"polar angle $\theta$ resolution(degrees)")
    axs1[1].set_ylabel("Count")
    axs1[1].set_xlim(-6, 6)

    axs1[1].set_title("p, theta, and phi resolution MC\n\n", loc='center', fontsize=24)


    axs1[2].hist(phi_true - phi_recon, bins=200, color="green")
    axs1[2].set_xlabel("azimuthal angle $\phi$ resolution(degrees)")
    axs1[2].set_ylabel("Count")

    fig1.savefig('fig0_delta_ptp.pdf')

    # Now reset truth arrays to test subset instead of full dataset for matching algorithm below
    p_true     = p_true[split_indices[1]:]
    theta_true = theta_true[split_indices[1]:]
    phi_true   = phi_true[split_indices[1]:]
    vx_true    = vx_true[split_indices[1]:]
    vy_true    = vy_true[split_indices[1]:]
    vz_true    = vz_true[split_indices[1]:]

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
    axs1[0,1].set_xlabel("polar angle (degrees)")
    axs1[0,1].set_ylabel("Count")

    axs1[0,1].set_title("p, theta, and phi normalized predictions\n\n", loc='center', fontsize=24)

    axs1[0,2].hist(phi_predictions_normalized, bins=50, color="peru")
    axs1[0,2].set_xlabel("azimuthal angle (degrees)")
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

    # Define vertex normalization parameters
    vx_max = 20 # max(vx) #NOTE: NOW JUST GUESS AT MAXES SO IT ISN'T DATASET OR FILE DEPENDENT AND YOU CAN THEN JUST LOAD BACK IN AND UNNORMALIZE NO PROBLEM.
    vy_max = 5  # max(vy)
    vz_max = 5  # max(vz)

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
    axs1[0,1].set_xlabel("polar angle (degrees)")
    axs1[0,1].set_ylabel("Count")

    axs1[0,1].set_title("p, theta, and phi predictions\n\n", loc='center', fontsize=24)

    axs1[0,2].hist(phi_predictions, bins=50, color="peru")
    axs1[0,2].set_xlabel("azimuthal angle (degrees)")
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

    # print("p vector lengths:", len(p_true), len(p_predictions), (len(p_true) * len(p_predictions)))

    # matching with cutoffs |theta_mc-theta_rec|<2deg and |phi_mc-phi_rec|<6deg, minimize quadrature sum error
    # change matching formula to also include momentum? |p_true - p_predictions|<0.05 GeV

    """
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
    """

    print("DEBUGGING: np.shape(p_predictions)     = ",np.shape(p_predictions))
    print("DEBUGGING: np.shape(p_true)            = ",np.shape(p_true))
    print("DEBUGGING: np.shape(theta_predictions) = ",np.shape(theta_predictions))
    print("DEBUGGING: np.shape(theta_true)        = ",np.shape(theta_true))
    print("DEBUGGING: np.shape(phi_predictions)   = ",np.shape(phi_predictions))
    print("DEBUGGING: np.shape(phi_true)          = ",np.shape(phi_true))
    print("DEBUGGING: np.shape(vx_predictions)    = ",np.shape(vx_predictions))
    print("DEBUGGING: np.shape(vx_true)           = ",np.shape(vx_true))
    print("DEBUGGING: np.shape(vy_predictions)    = ",np.shape(vy_predictions))
    print("DEBUGGING: np.shape(vy_true)           = ",np.shape(vy_true))
    print("DEBUGGING: np.shape(vz_predictions)    = ",np.shape(vz_predictions))
    print("DEBUGGING: np.shape(vz_true)           = ",np.shape(vz_true))

    p_difference = np.array(p_predictions) - np.array(p_true)
    theta_difference = np.array(theta_predictions) - np.array(theta_true)
    phi_difference = np.array(phi_predictions) - np.array(phi_true)
    vx_difference = np.array(vx_predictions) - np.array(vx_true)
    vy_difference = np.array(vy_predictions) - np.array(vy_true)
    vz_difference = np.array(vz_predictions) - np.array(vz_true)

    # Reset recon arrays to align with the test subset
    p_recon     = np.squeeze(xyz_test[:,-3]) * 2
    theta_recon = np.squeeze(xyz_test[:,-2]) * 180
    phi_recon   = np.squeeze(xyz_test[:,-1]) * 360 - 180

    p_difference_recon = np.array(p_recon) - np.array(p_true)
    theta_difference_recon = np.array(theta_recon) - np.array(theta_true)
    phi_difference_recon = np.array(phi_recon) - np.array(phi_true)
    #vx_difference_recon = np.array(vx_recon) - np.array(vx_true) #TODO: GET V*_RECON???
    #vy_difference_recon = np.array(vy_recon) - np.array(vy_true)
    #vz_difference_recon = np.array(vz_recon) - np.array(vz_true)

    # Create a grid of subplots
    fig1, axs1 = plt.subplots(2, 3, figsize=(25, 12))

    # Adjust spacing
    fig1.subplots_adjust(hspace=0.5)
    fig1.subplots_adjust(wspace=0.5)

    # predicted data resolution histograms

    axs1[0,0].hist(p_difference, bins=100, color="darkseagreen")
    axs1[0,0].set_ylabel("Count")
    axs1[0,0].set_xlabel(r"$\Delta p$ (GeV)")
    axs1[0,0].set_xlim(-0.1, 0.1)

    axs1[0,1].hist(theta_difference, bins=100, color="fuchsia")
    axs1[0,1].set_ylabel("Count")
    axs1[0,1].set_xlabel(r"$\Delta\theta$ (deg.)")
    axs1[0,1].set_xlim(-2, 2)

    axs1[0,2].hist(phi_difference, bins=3000, color="mediumblue")
    axs1[0,2].set_ylabel("Count")
    axs1[0,2].set_xlabel(r"$\Delta\phi$ (deg.)")
    axs1[0,2].set_xlim(-6, 6)

    axs1[1,0].hist(p_difference_recon, bins=100, color="darkseagreen")
    axs1[1,0].set_ylabel("Count")
    axs1[1,0].set_xlabel(r"$\Delta p$ (GeV)")
    axs1[1,0].set_xlim(-0.1, 0.1)

    axs1[1,1].hist(theta_difference_recon, bins=100, color="fuchsia")
    axs1[1,1].set_ylabel("Count")
    axs1[1,1].set_xlabel(r"$\Delta\theta$ (deg.)")
    axs1[1,1].set_xlim(-2, 2)

    axs1[1,2].hist(phi_difference_recon, bins=3000, color="mediumblue")
    axs1[1,2].set_ylabel("Count")
    axs1[1,2].set_xlabel(r"$\Delta\phi$ (deg.)")
    axs1[1,2].set_xlim(-6, 6)

    axs1[1,1].set_title("Base angle and momentum resolution", loc='center', fontsize=24)

    plt.suptitle("Predicted angle and momentum resolution", fontsize=24)

    # plt.show()
    fig1.savefig('fig3.pdf')

    return

# ------------------------------ END MAIN DEFINITION ------------------------------ #

#------------------------------ MAIN ------------------------------#
if __name__=="__main__":
    if len(sys.argv)<=2:
        print(
        "Usage: python3",os.path.abspath(sys.argv[0]),
        " 'xyz_regex' 'ptp_regex' "
        )
        print("NOTE: quotes around regex are important!")
        sys.exit(0)

    xyz_file_list = sorted([os.path.abspath(el) for el in glob.glob(sys.argv[1].replace(r"'",""))]) #NOTE: SORTING RELIES ON THE XYZ AND PTP FILES HAVING THE SAME NAMING SCHEMES
    ptp_file_list = sorted([os.path.abspath(el) for el in glob.glob(sys.argv[2].replace(r"'",""))])
    split         = (0.8,0.1,0.1) #NOTE: #TODO: SET FROM CLI
    main(
        xyz_file_list,
        ptp_file_list,
        split
        )
