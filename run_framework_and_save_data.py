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

def main(
    file_list,
    out_path1,
    out_path2,
    step=100):

    # Set LaTeX font for Matplotlib
    rc('text', usetex=True)

    # Set default plot font size
    plt.rcParams['font.size'] = 16

    # opening up some .hipo files

    #NOTE: COMMENTED OUT IN FAVOR OF USING HIPOPY.HIPOPY.ITERATE
    # filename0 = "/volatile/clas12/users/mfmce/mc_jobs_rga_vtx_3_23_23/skim_50nA_OB_job_3313_0.hipo"
    # filename1 = "/volatile/clas12/users/mfmce/mc_jobs_rga_vtx_3_23_23/skim_50nA_OB_job_3313_1.hipo"
    # filename2 = "/volatile/clas12/users/mfmce/mc_jobs_rga_vtx_3_23_23/skim_50nA_OB_job_3313_2.hipo"
    # filename3 = "/volatile/clas12/users/mfmce/mc_jobs_rga_vtx_3_23_23/skim_50nA_OB_job_3313_3.hipo"
    # filename4 = "/volatile/clas12/users/mfmce/mc_jobs_rga_vtx_3_23_23/skim_50nA_OB_job_3313_4.hipo"

    # hipo_file0 = hipopy.open(filename0,mode='r')
    # hipo_file1 = hipopy.open(filename1,mode='r')
    # hipo_file2 = hipopy.open(filename2,mode='r')
    # hipo_file3 = hipopy.open(filename3,mode='r')
    # hipo_file4 = hipopy.open(filename4,mode='r')

    # hipo_array = [hipo_file0, hipo_file1, hipo_file2, hipo_file3, hipo_file4]

    # # Showing the data types inside each bank

    # print("REC::Particle data","\n"+100*"=")
    # display(hipo_file1.getNamesAndTypes('REC::Particle'))
    # print(100*"=","\nREC::Traj data","\n"+100*"=")
    # display(hipo_file1.getNamesAndTypes('REC::Traj'))
    # print("MC::Lund data","\n"+100*"=")
    # display(hipo_file1.getNamesAndTypes('MC::Lund'))
    #NOTE: COMMENTED OUT ABOVE IN FAVOR OF USING HIPOPY.HIPOPY.ITERATE

    # defining function to find xF
    # q is the pion, p is the virtual photon
    def calculate_xF(q, p, init_target):
        W = (init_target + p).M()
        com = q + init_target
        comBOOST = com.BoostVector()
        qq = q
        pp = p
        qq.Boost(-comBOOST)
        pp.Boost(-comBOOST)
        return 2 * ( qq.Vect().Dot(pp.Vect()) ) / ( qq.Vect().Mag() * W )

    # defining functions to get Q2

    # for the pion
    def cthfunc(Px, Py, Pz):

        Pt = np.sqrt(Px*Px + Py*Py)
        return Pz / np.sqrt(Pz * Pz + Pt * Pt)

    # for the electrons
    def Q2func(E2, cth):
        E1 = 10.6 # GeV
        return 2 * E1 * E2 * (1.0 - cth)

    # in MC::Lund, we want the energy of the second electron, so we need this function:
    def find_second_index(lst, value):
        try:
            first_index = lst.index(value)
            return lst.index(value, first_index + 1)
        except ValueError:
            # Handle the case when the value is not found or there is no second occurrence
            return None

    def get_random_color():
        # Generate random values for red, green, and blue channels
        r = random.random()
        g = random.random()
        b = random.random()

        # Return the color as a tuple
        return r, g, b

    Mpion = 0.139570 # GeV
    Mproton = 0.938272 # GeV

    p_recon = []
    theta_recon = []
    phi_recon = []

    x = []
    y = []
    z = []

    p_true = []
    theta_true = []
    phi_true = []

    vx = []
    vy = []
    vz = []

    xF = []
    q2 = []

    # limit on iterations (for each hipo file)
    max_batches = 1000 #NOTE: MAXIMUM NUMBER OF BATCHES TO PROCESS FOR DEBUGGING LIMITATIONS...

    start_time = timeit.default_timer()
    banks = ["REC::Particle","REC::Traj","MC::Lund"]
    for k,batch in enumerate(hp.iterate(file_list,banks=banks,step=step)):
        if (k > max_batches):
                break
        
        px_array = batch["REC::Particle_px"]
        for i,_ in enumerate(px_array):
            # if (i > Nevents):
            #     break

            # pulling out data from each event
            detectors_temp = batch["REC::Traj_detector"][i]
            p_index_temp = batch["REC::Traj_pindex"][i]
            charge_temp = batch["REC::Particle_charge"][i]
            
            px_recon = batch["REC::Particle_px"][i]
            py_recon = batch["REC::Particle_py"][i]
            pz_recon = batch["REC::Particle_pz"][i]
            pid_recon = batch["REC::Particle_pid"][i]
                    
            # delete the data of the neutral particles since it'll make counting with pindex possible
            if 0 in charge_temp:
                first_zero_index = charge_temp.index(0)
                del px_recon[first_zero_index:], py_recon[first_zero_index:], pz_recon[first_zero_index:], pid_recon[first_zero_index:], charge_temp[first_zero_index:]
                
            pion_indices_recon = [q for q in range(len(pid_recon)) if pid_recon[q] == -211]
            
            # getting coordinate data from REC::Traj
            x_temp = batch["REC::Traj_x"][i]
            y_temp = batch["REC::Traj_y"][i]
            z_temp = batch["REC::Traj_z"][i]
            
            
            # getting info on the virtual photon so I can do the xF > 0 cut
            pid_Lund = batch["MC::Lund_pid"][i]
            px_Lund = batch["MC::Lund_px"][i]
            py_Lund = batch["MC::Lund_py"][i]
            pz_Lund = batch["MC::Lund_pz"][i]
            masses = batch["MC::Lund_mass"][i]
            energies = batch["MC::Lund_energy"][i]
            
            virtual_photon_index = pid_Lund.index(22)
            virtual_photon = ROOT.TLorentzVector()
            virtual_photon.SetXYZM(px_Lund[virtual_photon_index], py_Lund[virtual_photon_index], pz_Lund[virtual_photon_index], masses[virtual_photon_index])
            
            # initial target: proton at rest
            proton_target = ROOT.TLorentzVector()
            proton_target.SetXYZM(0, 0, 0, Mproton)
            
            # Q2 cut
            scattered_electron_index = find_second_index(pid_Lund, 11)
            cth = cthfunc(px_Lund[scattered_electron_index], py_Lund[scattered_electron_index], pz_Lund[scattered_electron_index])

            q2_temp = Q2func(energies[scattered_electron_index], cth)
            q2.append(q2_temp)
            if q2_temp < 1:
                continue
            
            # looping through reconstructed pions
            for j,_ in enumerate(pion_indices_recon):
                
                pion_location = pion_indices_recon[j]
                pion_vector_recon = ROOT.TLorentzVector()
                pion_vector_recon.SetXYZM(px_recon[pion_location], py_recon[pion_location], pz_recon[pion_location], Mpion)
                
                # xF cut
                xF_temp = calculate_xF(pion_vector_recon, virtual_photon, proton_target)
                xF.append(xF_temp)
                if xF_temp <= 0:
                    continue
                
                # we want the coordinate data between the first instance of pion_location and the last instance of pion_location
                first_instance = p_index_temp.index(pion_location)
                last_instance = len(p_index_temp) - p_index_temp[::-1].index(pion_location) - 1
                
                # getting coordinate input strings of ONLY 11 x, y, and z coordinates per pion
                if ((last_instance - first_instance) != 10):
                    continue
                x_append = []
                y_append = []
                z_append = []
                
                for m in range(first_instance, last_instance + 1):  # end value for range is exclusive
                    x_append.append(x_temp[m])
                    y_append.append(y_temp[m])
                    z_append.append(z_temp[m])
                
                
                
                p_recon_temp = pion_vector_recon.P()
                if (p_recon_temp < 2): # let's not deal with the pions with stupid high momentum
                    
                    p_recon.append(p_recon_temp) # append total momentum
                    theta_recon.append(pion_vector_recon.Theta() *180/np.pi) # append calculated azimuthal angle in degrees
                    phi_recon.append(pion_vector_recon.Phi() *180/np.pi) # append calculated polar angle in degrees
                    
                    x.append(x_append)
                    y.append(y_append)
                    z.append(z_append)
                    
            
            # true data now
            
            px_true = batch["MC::Lund_px"][i]
            py_true = batch["MC::Lund_py"][i]
            pz_true = batch["MC::Lund_pz"][i]
            pid_true = batch["MC::Lund_pid"][i]
            
            pion_indices_true = [q for q in range(len(pid_true)) if pid_true[q] == -211]
            
            # looping through pions
            for j,_ in enumerate(pion_indices_true):
                
                pion_location = pion_indices_true[j]
                pion_vector_true = ROOT.TLorentzVector()
                pion_vector_true.SetXYZM(px_true[pion_location], py_true[pion_location], pz_true[pion_location], Mpion)
                
                # xF cut
                xF_temp = calculate_xF(pion_vector_true, virtual_photon, proton_target)
                xF.append(xF_temp)
                if xF_temp <= 0:
                    continue
                
                p_true.append(pion_vector_true.P()) # append total momentum
                theta_true.append(pion_vector_true.Theta() * 180 / np.pi) # append calculated azimuthal angle in degrees
                phi_true.append(pion_vector_true.Phi() * 180 / np.pi) # append calculated polar angle in degrees
                
                vx.append(batch["MC::Lund_vx"][i][pion_location])
                vy.append(batch["MC::Lund_vy"][i][pion_location])
                vz.append(batch["MC::Lund_vz"][i][pion_location])
            
    print("time spent computing:", timeit.default_timer() - start_time)

        
    print(vx[:25], "\n\n", vy[:25], "\n\n", vz[:25])


    print("p vector lengths:", len(p_true), len(p_recon))

    # matching with cutoffs |theta_mc-theta_rec|<2deg and |phi_mc-phi_rec|<6deg, minimize quadrature sum error
    # change matching formula to also include momentum? |p_true - p_recon|<0.05 GeV

    start_time = timeit.default_timer()

    p_recon_changing = p_recon[:]
    theta_recon_changing = theta_recon[:]
    phi_recon_changing = phi_recon[:]

    matching_indices = np.zeros((2, len(p_true)), dtype=int)
    unmatched_indices = np.array([], dtype=int)
    num_matches = 0

    # going through all the "true" pions
    for i,_ in enumerate(p_true):
        
        min_quad_error = 1000 # arbitrary large number
        for j,_ in enumerate(p_recon): # for each true pion, look through every reconstructed pion
            
            theta_error = theta_true[i] - theta_recon_changing[j]
            phi_error = phi_true[i] - phi_recon_changing[j]
            p_error = p_true[i] - p_recon_changing[j] 
            
            if ( (abs(theta_error) < 2 or abs(theta_error) > 178) & (abs(phi_error) < 6 or abs(phi_error) > 354) & (abs(p_error) < 0.05) ): # does this true pion have a similar reconstructed pion, based on their angles?
                min_quad_error_temp = np.sqrt((theta_error) ** 2 + (phi_error) ** 2) + 10 * p_error # in case there are multiple hits, we can minimize based off a quadrature sum
                if (min_quad_error_temp < min_quad_error):
                    min_quad_error = min_quad_error_temp
                    recon_index = j # saving the matched index of the "best" reconstructed pion
                    
        if (min_quad_error < 1000): # if there were any matches for this true pion, essentially
            matching_indices[0][num_matches] = i
            matching_indices[1][num_matches] = recon_index # saving the matching indices of the true and reconstructed pions
            
            p_recon_changing[recon_index] = 100
            theta_recon_changing[recon_index] = 100
            phi_recon_changing[recon_index] = 100  # this is my way of avoiding duplicates - obviously angles of 100 degrees will not be put as a match to other pions going forward
            
            num_matches += 1
        else:
            unmatched_indices = np.append(unmatched_indices, [i])
            
    matching_indices = matching_indices[:, :num_matches]

    print("time computing", timeit.default_timer() - start_time)
    print(matching_indices)
    print(len(p_true) * len(p_recon))

    print(num_matches, len(p_true), len(p_recon))

    # rearrange data to input better

    p_true_constant = p_true[:]
    p_recon_constant = p_recon[:]

    theta_true_constant = theta_true[:]
    theta_recon_constant = theta_recon[:]

    phi_true_constant = phi_true[:]
    phi_recon_constant = phi_recon[:]

    x_constant = x[:]
    y_constant = y[:]
    z_constant = z[:]

    vx_constant = vx[:]
    vy_constant = vy[:]
    vz_constant = vz[:]

    for i in range(0, len(matching_indices[0])):
        p_true[i] = p_true_constant[matching_indices[0][i]]
        p_recon[i] = p_recon_constant[matching_indices[1][i]]
        theta_true[i] = theta_true_constant[matching_indices[0][i]]
        theta_recon[i] = theta_recon_constant[matching_indices[1][i]]
        phi_true[i] = phi_true_constant[matching_indices[0][i]]
        phi_recon[i] = phi_recon_constant[matching_indices[1][i]]
        
        x[i] = x_constant[matching_indices[1][i]]
        y[i] = y_constant[matching_indices[1][i]]
        z[i] = z_constant[matching_indices[1][i]]
        
        vx[i] = vx_constant[matching_indices[0][i]]
        vy[i] = vy_constant[matching_indices[0][i]]
        vz[i] = vz_constant[matching_indices[0][i]]

    print(np.array(p_true[:25]) - np.array(p_recon[:25]))

    # shortening data to the good stuff

    p_true = p_true[:num_matches]
    p_recon = p_recon[:num_matches]
    theta_true = theta_true[:num_matches]
    theta_recon = theta_recon[:num_matches]
    phi_true = phi_true[:num_matches]
    phi_recon = phi_recon[:num_matches]

    x = x[:num_matches]
    y = y[:num_matches]
    z = z[:num_matches]

    vx = vx[:num_matches]
    vy = vy[:num_matches]
    vz = vz[:num_matches]

    # Create a grid of subplots
    fig1, axs1 = plt.subplots(1, 3, figsize=(25, 8))

    # Adjust spacing
    fig1.subplots_adjust(hspace=0.5)
    fig1.subplots_adjust(wspace=0.5)

    axs1[0].hist(np.array(p_true) - np.array(p_recon), bins=200, color="black")
    axs1[0].set_xlabel("momentum, p resolution(GeV)")
    axs1[0].set_ylabel("Count")

    axs1[1].hist(np.array(theta_true) - np.array(theta_recon), bins=200, color="blue")
    axs1[1].set_xlabel(r"azimuthal angle $\theta$ resolution(degrees)")
    axs1[1].set_ylabel("Count")

    axs1[1].set_title("p, theta, and phi resolution MC\n\n", loc='center', fontsize=24)


    axs1[2].hist(np.array(phi_true) - np.array(phi_true), bins=200, color="green")
    axs1[2].set_xlabel("polar angle $\phi$ resolution(degrees)")
    axs1[2].set_ylabel("Count")

    fig1.savefig('test2.pdf')
    # plt.show()

    # Create a grid of subplots
    fig1, axs1 = plt.subplots(2, 3, figsize=(25, 16))

    # Adjust spacing
    fig1.subplots_adjust(hspace=0.5)
    fig1.subplots_adjust(wspace=0.5)

    axs1[0,0].hist(p_true, bins=200, color="black")
    axs1[0,0].set_xlabel("momentum, p (GeV)")
    axs1[0,0].set_ylabel("Count")

    axs1[0,1].hist(theta_true, bins=200, color="blue")
    axs1[0,1].set_xlabel(r"azimuthal angle $\theta$ (degrees)")
    axs1[0,1].set_ylabel("Count")

    axs1[0,1].set_title("p, theta, and phi true MC\n\n", loc='center', fontsize=24)

    axs1[0,2].hist(phi_true, bins=200, color="green")
    axs1[0,2].set_xlabel("polar angle $\phi$ (degrees)")
    axs1[0,2].set_ylabel("Count")


    axs1[1,0].hist(vx, bins=200, color="black")
    axs1[1,0].set_xlabel("vx (cm)")
    axs1[1,0].set_ylabel("Count")

    axs1[1,1].hist(vy, bins=200, color="blue")
    axs1[1,1].set_xlabel("vy (cm)")
    axs1[1,1].set_ylabel("Count")

    axs1[1,1].set_title("vertex true MC\n\n", loc='center', fontsize=24)

    axs1[1,2].hist(np.array(vz)[np.array(vz) < 20], bins=200, color="green")
    axs1[1,2].set_xlabel("vz (cm)")
    axs1[1,2].set_ylabel("Count")

    fig1.savefig('test3.pdf')
    # plt.show()

    def normalize_2d_list(data):
        # Find the global minimum and maximum values for all variables
        flat_data = [item for sublist in data for item in sublist]
        global_min = min(flat_data)
        global_max = max(flat_data)

        # Normalize the data
        normalized_data = [[(value - global_min) / (global_max - global_min) for value in row] for row in data]

        return normalized_data

    # normalize data

    # training with MC data

    # normalize input data
    x_normalized = normalize_2d_list(x)
    y_normalized = normalize_2d_list(y)
    z_normalized = normalize_2d_list(z)

    # # normalize vertex output data
    vx_max = 20 # max(vx) #NOTE: NOW JUST GUESS AT MAXES SO IT ISN'T DATASET OR FILE DEPENDENT AND YOU CAN THEN JUST LOAD BACK IN AND UNNORMALIZE NO PROBLEM.
    vy_max = 5  # max(vy)
    vz_max = 5  # max(vz)

    vx_normalized = [filler / vx_max for filler in vx] 
    vy_normalized = [filler / vy_max for filler in vy]
    vz_normalized = [filler / vz_max for filler in vz]


    # Normalize theta values to the range [0, 1]
    theta_true_normalized = [theta / 180 for theta in theta_true]

    # Normalize phi values to the range [0, 1]
    phi_true_normalized = [(phi + 180) / 360 for phi in phi_true]

    # Normalize p values to the range [0, 1]
    p_true_normalized = [p / 2 for p in p_true]

    # RECON
    # Normalize theta values to the range [0, 1]
    theta_recon_normalized = [theta / 180 for theta in theta_recon]

    # Normalize phi values to the range [0, 1]
    phi_recon_normalized = [(phi + 180) / 360 for phi in phi_recon]

    # Normalize p values to the range [0, 1]
    p_recon_normalized = [p / 2 for p in p_recon]

    # Create a grid of subplots
    fig1, axs1 = plt.subplots(2, 3, figsize=(25, 16))

    # Adjust spacing
    fig1.subplots_adjust(hspace=0.5)
    fig1.subplots_adjust(wspace=0.5)

    axs1[0,0].hist(p_true_normalized, bins=200, color="black")
    axs1[0,0].set_xlabel("momentum, p (GeV)")
    axs1[0,0].set_ylabel("Count")

    axs1[0,1].hist(theta_true_normalized, bins=200, color="blue")
    axs1[0,1].set_xlabel("azimuthal angle (degrees)")
    axs1[0,1].set_ylabel("Count")

    axs1[0,1].set_title("p, theta, and phi normalized\n\n", loc='center', fontsize=24)

    axs1[0,2].hist(phi_true_normalized, bins=200, color="green")
    axs1[0,2].set_xlabel("polar angle (degrees)")
    axs1[0,2].set_ylabel("Count")


    axs1[1,0].hist(vx_normalized, bins=200, color="black")
    axs1[1,0].set_xlabel("vx")
    axs1[1,0].set_ylabel("Count")

    axs1[1,1].hist(vy_normalized, bins=200, color="blue")
    axs1[1,1].set_xlabel("vy")
    axs1[1,1].set_ylabel("Count")

    axs1[1,1].set_title("vertex normalized\n\n", loc='center', fontsize=24)

    axs1[1,2].hist(vz_normalized, bins=200, color="green")
    axs1[1,2].set_xlabel("vz")
    axs1[1,2].set_ylabel("Count")

    fig1.savefig('test1.pdf')
    # plt.show()

    # splitting data into training data (70%), validation data (10%), and testing data (20%)

    split_index_1_i = int(len(x_normalized) * 0.7)
    split_index_2_i = int(len(x_normalized) * 0.8)

    split_index_1_o = int(len(p_true_normalized) * 0.7)
    split_index_2_o = int(len(p_true_normalized) * 0.8)

    x_training = x_normalized[:split_index_1_i]
    x_validation = x_normalized[split_index_1_i:split_index_2_i]
    x_test = x_normalized[split_index_2_i:]
    y_training = y_normalized[:split_index_1_i]
    y_validation = y_normalized[split_index_1_i:split_index_2_i]
    y_test = y_normalized[split_index_2_i:]
    z_training = z_normalized[:split_index_1_i]
    z_validation = z_normalized[split_index_1_i:split_index_2_i]
    z_test = z_normalized[split_index_2_i:]

    p_input_training = [[item] for item in p_recon_normalized[:split_index_1_i]]
    p_input_validation = [[item] for item in p_recon_normalized[split_index_1_i:split_index_2_i]]
    p_input_test = [[item] for item in p_recon_normalized[split_index_2_i:]]
    theta_input_training = [[item] for item in theta_recon_normalized[:split_index_1_i]]
    theta_input_validation = [[item] for item in theta_recon_normalized[split_index_1_i:split_index_2_i]]
    theta_input_test = [[item] for item in theta_recon_normalized[split_index_2_i:]]
    phi_input_training = [[item] for item in phi_recon_normalized[:split_index_1_i]]
    phi_input_validation = [[item] for item in phi_recon_normalized[split_index_1_i:split_index_2_i]]
    phi_input_test = [[item] for item in phi_recon_normalized[split_index_2_i:]]

    p_training = p_true_normalized[:split_index_1_o]
    p_validation = p_true_normalized[split_index_1_o:split_index_2_o]
    p_test = p_true_normalized[split_index_2_o:]
    theta_training = theta_true_normalized[:split_index_1_o]
    theta_validation = theta_true_normalized[split_index_1_o:split_index_2_o]
    theta_test = theta_true_normalized[split_index_2_o:]
    phi_training = phi_true_normalized[:split_index_1_o]
    phi_validation = phi_true_normalized[split_index_1_o:split_index_2_o]
    phi_test = phi_true_normalized[split_index_2_o:]

    vx_training = vx_normalized[:split_index_1_o]
    vx_validation = vx_normalized[split_index_1_o:split_index_2_o]
    vx_test = vx_normalized[split_index_2_o:]
    vy_training = vy_normalized[:split_index_1_o]
    vy_validation = vy_normalized[split_index_1_o:split_index_2_o]
    vy_test = vy_normalized[split_index_2_o:]
    vz_training = vz_normalized[:split_index_1_o]
    vz_validation = vz_normalized[split_index_1_o:split_index_2_o]
    vz_test = vz_normalized[split_index_2_o:]

    # input for training the neural network
    xyz_training = [sublist1 + sublist2 + sublist3 + sublist4 + sublist5 + sublist6 for sublist1, sublist2, sublist3, sublist4, sublist5, sublist6
                    in zip(x_training, y_training, z_training, p_input_training, theta_input_training, phi_input_training)]  

    # output labels for training the net
    # ALSO includes vertex data now
    p_theta_phi_training = [[0,0,0,0,0,0] for i in range(0, len(p_training))]
    for i,_ in enumerate(p_training):
        p_theta_phi_training[i][0] = p_training[i]
        p_theta_phi_training[i][1] = theta_training[i]
        p_theta_phi_training[i][2] = phi_training[i]
        p_theta_phi_training[i][3] = vx_training[i]
        p_theta_phi_training[i][4] = vy_training[i]
        p_theta_phi_training[i][5] = vz_training[i]

    # validation input data
    xyz_validation = [sublist1 + sublist2 + sublist3 + sublist4 + sublist5 + sublist6 for sublist1, sublist2, sublist3, sublist4, sublist5, sublist6
                    in zip(x_validation, y_validation, z_validation, p_input_validation, theta_input_validation, phi_input_validation)]  
    # validation output data
    p_theta_phi_validation = [[0,0,0,0,0,0] for i in range(0, len(p_validation))]
    for i,_ in enumerate(p_validation):
        p_theta_phi_validation[i][0] = p_validation[i]
        p_theta_phi_validation[i][1] = theta_validation[i]
        p_theta_phi_validation[i][2] = phi_validation[i]
        p_theta_phi_validation[i][3] = vx_validation[i]
        p_theta_phi_validation[i][4] = vy_validation[i]
        p_theta_phi_validation[i][5] = vz_validation[i]

    # test input data
    xyz_test = [sublist1 + sublist2 + sublist3 + sublist4 + sublist5 + sublist6 for sublist1, sublist2, sublist3, sublist4, sublist5, sublist6
                    in zip(x_test, y_test, z_test, p_input_test, theta_input_test, phi_input_test)]  
    # test output data
    p_theta_phi_test = [[0,0,0,0,0,0] for i in range(0, len(p_test))]
    for i,_ in enumerate(p_test):
        p_theta_phi_test[i][0] = p_test[i]
        p_theta_phi_test[i][1] = theta_test[i]
        p_theta_phi_test[i][2] = phi_test[i]
        p_theta_phi_test[i][3] = vx_test[i]
        p_theta_phi_test[i][4] = vy_test[i]
        p_theta_phi_test[i][5] = vz_test[i]


    # Convert training input data to a list of arrays
    xyz_training = [np.array(sublist, dtype=float) for sublist in xyz_training]

    # Convert training output labels to a list of arrays
    p_theta_phi_training = [np.array(sublist, dtype=float) for sublist in p_theta_phi_training]

    # Convert validation input data to a list of arrays
    xyz_validation = [np.array(sublist, dtype=float) for sublist in xyz_validation]

    # Convert validation output labels to a list of arrays
    p_theta_phi_validation = [np.array(sublist, dtype=float) for sublist in p_theta_phi_validation]

    # Convert test input data to a list of arrays
    xyz_test = [np.array(sublist, dtype=float) for sublist in xyz_test]

    # Convert test output labels to a list of arrays
    p_theta_phi_test = [np.array(sublist, dtype=float) for sublist in p_theta_phi_test]

    # Find the maximum sequence length
    max_sequence_length = max(len(seq) for seq in xyz_training + xyz_validation + xyz_test)

    # Pad sequences to the maximum length

    xyz_training = pad_sequences(xyz_training, maxlen=max_sequence_length, dtype=float)
    xyz_validation = pad_sequences(xyz_validation, maxlen=max_sequence_length, dtype=float)
    xyz_test = pad_sequences(xyz_test, maxlen=max_sequence_length, dtype=float)

    print(len(xyz_training), len(xyz_training[0]), len(p_theta_phi_training), len(p_theta_phi_training[0]))

    #NOTE: ADDED: SAVE ENTIRE DATASET TO CSV
    xyz_full = np.concatenate((xyz_training,xyz_validation,xyz_test))
    ptp_full = np.concatenate((p_theta_phi_training,p_theta_phi_validation,p_theta_phi_test))
    np.savetxt(out_path1, xyz_full, delimiter=',')
    np.savetxt(out_path2, ptp_full, delimiter=',')

    print("EXITING MAIN")

#------------------------------ MAIN ------------------------------#
if __name__=="__main__":
    if len(sys.argv)<=3:
        print("Usage: python3 ",os.path.abspath(sys.argv[0])," outpath_xyz outpath_ptp file1 file2 file3 ...")
        sys.exit(0)
    out_path1 = os.path.abspath(sys.argv[1])
    out_path2 = os.path.abspath(sys.argv[2])
    file_list = sys.argv[3:]
    main(
        file_list,
        out_path1,
        out_path2,
    )
