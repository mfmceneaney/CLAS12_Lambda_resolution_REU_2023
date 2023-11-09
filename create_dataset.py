#----------------------------------------------------------------------#
# Slow pion dataset creation script

# Data
import numpy as np
import awkward as ak
import pandas as pd

# I/O
import uproot as ur
import hipopy.hipopy as hp # <--- Package for reading in the hipo files

# Plotting
import matplotlib.pyplot as plt

# Physics
from particle import PDGID

# Miscellaneous
import os
import sys #NOTE: ADDED
import tqdm

#----------------------------------------------------------------------#
# HIPO bank reading and linking functions

def get_bank_keys(bank_name,all_keys,separator='_'):
    """
    :description: Get list of the keys for given bank name from a list of all batch keys.
    
    :param: bank_name
    :param: all_keys
    :param: separator='_'
    
    :return: bank_keys
    """
    bank_keys = []
    for key in all_keys:
        if key.startswith(bank_name+separator):
            bank_keys.append(key)
    return bank_keys
        
def get_event_table(bank_keys,event_num,batch,dtype=float):
    """
    :description: Get a bank event table as a numpy array of shape (number of columns, number of rows).
    
    :param: bank_keys
    :param: event_num
    :param: batch
    :param: dtype=float
    
    :return: bank_table
    """
    bank_table = []
    bank_table = np.moveaxis(np.array([batch[key][event_num] for key in bank_keys], dtype=dtype),[0,1],[1,0])
    return bank_table

def get_link_indices(event_table_rec_particle,event_table,pindex_idx=1):
    """
    :description: Get index pairs linking entries in a bank back to entries in the 'REC::Particle' bank.
    
    :param: event_table_rec_particle
    :param: event_table
    :param: pindex_idx=1
    
    :return: link_indices
    """
    
    link_indices = []
    nrec = np.shape(event_table_rec_particle)[0]
    for rec_particle_idx in range(0,nrec):
        for event_table_idx, el in enumerate(event_table[:,pindex_idx]):
            if el==rec_particle_idx:
                link_indices.append([rec_particle_idx,event_table_idx])
    return np.array(link_indices,dtype=int) #NOTE: link_indices = [(event_table_idx,rec_particle_idx)]

def get_parent_indices(mc_event_table,index_idx=0,parent_idx=4,daughter_idx=5):
    """
    TODO
    """
    for mc_event_table_idx, index in enumerate(mc_event_table[:,index_idx]):
        pass
    pass

def get_match_indices(
    rec_event_table,
    mc_event_table,
    rec_px_idx             = 1,
    rec_py_idx             = 2,
    rec_pz_idx             = 3,
    rec_ch_idx             = 8,
    mc_px_idx              = 6,
    mc_py_idx              = 7,
    mc_pz_idx              = 8,
    mc_pid_idx             = 3,
    mc_daughter_idx        = 5,
    match_charge           = True,
    require_no_mc_daughter = True,
    enforce_uniqueness     = True,
    ):
    """
    :description: Get index pairs matching 
    
    :param: rec_event_table
    :param: mc_event_table
    :param: rec_px_idx             = 1,
    :param: rec_py_idx             = 2,
    :param: rec_pz_idx             = 3,
    :param: rec_ch_idx             = 8,
    :param: mc_px_idx              = 6,
    :param: mc_py_idx              = 7,
    :param: mc_pz_idx              = 8,
    :param: mc_pid_idx             = 3,
    :param: mc_daughter_idx        = 5,
    :param: match_charge           = True,
    :param: require_no_mc_daughter = True,
    :param: enforce_uniqueness     = True,
    
    :return: match_indices
    """
    
    # Set minimum
    rec_final_state_min_idx = 1
    mc_final_state_min_idx  = 3 #NOTE: MC::Lund bank is structured [e, p, q, e', all the other final state particles...]
    
    # Initialize index map
    match_indices    = -np.ones((rec_event_table.shape[0],2),dtype=float)
    match_indices[0] = [0,3] #NOTE: Always match first entry in REC::Particle to scattered electron in MC::Lund.

    # Get REC::Particle info
    rec_px    = rec_event_table[:,rec_px_idx]
    rec_py    = rec_event_table[:,rec_py_idx]
    rec_pz    = rec_event_table[:,rec_pz_idx]
    rec_pT    = np.sqrt(np.square(rec_event_table[:,rec_px_idx])+np.square(rec_event_table[:,rec_py_idx]))
    rec_p     = np.sqrt(np.square(rec_event_table[:,rec_px_idx])+np.square(rec_event_table[:,rec_py_idx])+np.square(rec_event_table[:,rec_pz_idx]))
    rec_theta = np.array(rec_pz)
    rec_theta = np.arctan(rec_pT,rec_theta)
    rec_phi   = np.arctan2(rec_py,rec_px)
    
    # Get MC::Lund info
    mc_px    = mc_event_table[:,mc_px_idx]
    mc_py    = mc_event_table[:,mc_py_idx]
    mc_pz    = mc_event_table[:,mc_pz_idx]
    mc_pT    = np.sqrt(np.square(mc_event_table[:,mc_px_idx])+np.square(mc_event_table[:,mc_py_idx]))
    mc_p     = np.sqrt(np.square(mc_event_table[:,mc_px_idx])+np.square(mc_event_table[:,mc_py_idx])+np.square(mc_event_table[:,mc_pz_idx]))
    mc_theta = np.array(mc_pz)
    mc_theta = np.arctan(mc_pT,mc_theta)
    mc_phi   = np.arctan2(mc_py,mc_px)

    # Loop rec particles
    for rec_idx, rec_part in enumerate(rec_event_table):
        
        # Start with final state particles past scattered electron
        if rec_idx<rec_final_state_min_idx: continue
        
        # Get REC::Particle charge
        rec_ch = rec_event_table[rec_idx,rec_ch_idx]
        
        # Loop mc particles
        mc_match_idx = -1
        min_domega   = 9999
        for mc_idx, mc_part in enumerate(mc_event_table):
            
            # Start with final state particles past scattered electron
            if mc_idx<mc_final_state_min_idx:
                continue
            
            # Enforce unique matching
            if enforce_uniqueness and mc_idx in match_indices[:,1]:
                continue
            
            # Match charge and require that the MC particle be final state (no daughters)
            if match_charge and rec_ch!=PDGID(mc_event_table[mc_idx,mc_pid_idx]).charge:
                continue
            if require_no_mc_daughter and mc_event_table[mc_idx,mc_daughter_idx]!=0:
                continue
                
            # Get angular and momentum differences
            dp     = np.abs(rec_p[rec_idx]     - mc_p[mc_idx])
            dtheta = np.abs(rec_theta[rec_idx] - mc_theta[mc_idx])
            dphi   = np.abs(rec_phi[rec_idx]   - mc_phi[mc_idx]) if np.abs(rec_phi[rec_idx] - mc_phi[mc_idx])<np.pi else 2*np.pi-np.abs(rec_phi[rec_idx] - mc_phi[mc_idx])
            domega = dp**2 + dtheta**2 + dphi**2
            
            # Reset angular, momentum minimum difference
            if domega<min_domega:
                min_domega   = domega
                mc_match_idx = mc_idx
                
        # Append matched index pair
        match_indices[rec_idx] = [rec_idx,mc_match_idx]
        
    return np.array(match_indices,dtype=int) #NOTE: IMPORTANT!

def get_info(base_indices,link_indices,bank_entry_indices,bank_event_table):
    """
    :description: Get selected entry info from other banks linked to REC::Particle.
    
    :param: base_indices
    :param: link_indices #NOTE: if None assume bank is REC::Particle and use identity map
    :param: bank_entry_indices
    :param: bank_event_table
    
    :return: bank_info as awkward.Array
    """
    if link_indices is None:
        bank_info = []
        for base_idx in base_indices:
            base_info = bank_event_table[base_idx,bank_entry_indices]
            bank_info.append([base_info])
            
        return ak.Array(bank_info)
            
    bank_info = []
    for base_idx in base_indices:
        base_info = []
        for rec_particle_idx, link_idx in link_indices:
            if rec_particle_idx==base_idx:
                base_info.append(bank_event_table[link_idx,bank_entry_indices]) #NOTE: INDICES HAVE TO BE INTEGERS...COULD ADD CHECK...
        if len(base_info)==0: #NOTE: Address case that no matches exist between banks
            base_info.append(np.zeros((len(bank_entry_indices),)))
        bank_info.append(base_info)
    
    return ak.Array(bank_info)

def get_truth_info(base_indices,match_indices,truth_entry_indices,mc_event_table):
    """
    :description: Get selected entry info from other banks linked to REC::Particle.
    
    :param: base_indices
    :param: link_indices #NOTE: if None assume bank is REC::Particle and use identity map
    :param: bank_entry_indices
    :param: bank_event_table
    
    :return: bank_info as awkward.Array
    """
    
    bank_info = []
    for base_idx in base_indices:
        base_info = []
        for rec_particle_idx, match_idx in match_indices:
            if rec_particle_idx==base_idx:
                base_info.append(mc_event_table[match_idx,truth_entry_indices]) #NOTE: INDICES HAVE TO BE INTEGERS...COULD ADD CHECK...
        if len(base_info)==0: #NOTE: Address case that no matches exist between banks
            base_info.append(np.zeros((len(truth_entry_indices),)))
        bank_info.append(base_info)
    
    return ak.Array(bank_info)

def remove_replacement_header(filename,replacement_header="REPLACEMENT_HEADER")
    # Read in the file
    with open(filename, 'r') as file:
        filedata = file.read()

    # Replace the target string
    filedata = filedata.replace('# '+replacement_header, '')

    # Write the file out again
    with open(filename, 'w') as file:
        file.write(filedata)

#----------------------------------------------------------------------#
# Main
def main(
    # Set input files, banks, and step size
    file_list = [
        '/volatile/clas12/users/mfmce/mc_jobs_rga_vtx_3_23_23/skim_50nA_OB_job_3313_0.hipo'
    ],
    banks = [
        'REC::Particle',
        'MC::Lund',
        'REC::Traj',
    ],
    step  = 11,

    # Set padding dimension (gets multiplied by len(rec_traj_keys))
    max_linked_entries = 30,

    # Set output files
    data_file_name  = "data.txt",
    truth_file_name = "truth.txt",
    delimiter       = ",",

    # Set entries to use as data/truth
    rec_particle_entry_indices = [1,2,3,4,5,6,7,9], # (pid), px, py, pz, vx, vy, vz, vt (Only in data), (charge), beta, (chi2pid), (status) #TODO: SET THESE OUTSIDE LOOPS
    rec_traj_entry_indices     = [2,3,4,5,6,7,8,9,10], # (pindex), (index), detector, layer, x, y, z, cx, cy, cz, path #TODO: SET THESE OUTSIDE LOOPS
    truth_entry_indices        = [11,12,13], # vx, vy, vz #TODO: SET THESE OUTSIDE LOOPS
    ):

    # Iterate hipo file
    for batch_num, batch in tqdm.tqdm(enumerate(hp.iterate(file_list,banks=banks,step=step))):
        
        # Set bank names and entry names to look at
        all_keys          = list(batch.keys())
        rec_particle_name = 'REC::Particle'
        rec_particle_keys = get_bank_keys(rec_particle_name,all_keys)
        mc_lund_name      = 'MC::Lund'
        mc_lund_keys      = get_bank_keys(mc_lund_name,all_keys)
        rec_traj_name     = 'REC::Traj'
        rec_traj_keys     = get_bank_keys(rec_traj_name,all_keys)
        
        # Loop events in batch
        batch_info  = None
        batch_truth = None
        for event_num, _ in enumerate(range(0,len(batch[list(batch.keys())[0]]))):
            
            # Get pion indices in REC::Particle (if they exist)
            filter_pid   = -211
            rec_pid_idx  = 0
            base_indices = np.where(np.array(batch['REC::Particle_pid'][event_num],dtype=object)==filter_pid)[0]
            if len(base_indices)<1: continue #NOTE: Check that particles of interest actually present in event
            
            # Get REC::Particle bank
            rec_particle_event_table = get_event_table(rec_particle_keys,event_num,batch,dtype=float)
            
            # Get MC::Lund bank and MC->REC matching indices
            mc_event_table = get_event_table(mc_lund_keys,event_num,batch,dtype=float)
            match_indices  = get_match_indices(rec_particle_event_table,mc_event_table)#TODO: This is somehow resetting rec_particle_event_table...
            if np.max(match_indices[:,-1])==-1: continue #NOTE: Check that particles actually get matched can also check that this is the case for base_indices...
            
            # Get REC::Traj bank and linking indices to REC::Particle
            rec_traj_event_table  = get_event_table(rec_traj_keys,event_num,batch,dtype=float)
            rec_traj_link_indices = get_link_indices(rec_particle_event_table,rec_traj_event_table,pindex_idx=1)
            
            # Get linked info for REC::Particle bank
            rec_particle_info = get_info(base_indices,None,rec_particle_entry_indices,rec_particle_event_table)
            
            # Get linked info for REC::Traj bank
            rec_traj_info     = get_info(base_indices,rec_traj_link_indices,rec_traj_entry_indices,rec_traj_event_table)
            
            # Get linked info for REC::Traj bank
            truth_info        = get_truth_info(base_indices,match_indices,truth_entry_indices,mc_event_table)
            
            #TODO: Could add optional computation of REC/MC kinematics assuming trigger particle is first...should add check for that though

            #----------------------------------------------------------------------#
            #NOTE: Have truth array of dim (nEvents,nPions,nTruth=nVtx=3 right now)
            #
            #----------------------------------------------------------------------#

            #----------------------------------------------------------------------#
            # TODO: Here have 3 options:
            # 1. Create pyg graphs EVENT-BY-EVENT                                      -> GNN
            # 2. Create awkward arrays of dim (nPions,nBanks(X)nEntries,nCols->PADDED) -> CNN
            # 3. Create awkward arrays of dim (nPions,nBanks(X)nEntries*nCols->PADDED) -> NN
            #----------------------------------------------------------------------#
            
            #NOTE: OPTION 3 ACTUAL: Create event data array of dim (nPions,nBanks(X)nEntries*nCols->PADDED=(1*8*1+1*9*25(PAD))=233)
            event_truth       = ak.to_numpy(ak.flatten(truth_info,axis=-1)) #NOTE: THESE NEED TO BE NUMPY ARRAYS TO WRITE WITH np.savetxt() BELOW
            rec_particle_info = ak.to_numpy(ak.flatten(rec_particle_info,axis=-1))
            rec_traj_info     = ak.flatten(rec_traj_info,axis=-1)
            targetD           = max_linked_entries*len(rec_traj_entry_indices)
            rec_traj_info     = np.nan_to_num(np.ma.array(ak.to_numpy(ak.pad_none(rec_traj_info,targetD,axis=-1))))#np.clip(np.nan_to_num(),1e-100,1e+100)
            rec_traj_info[rec_traj_info.mask] = 0.0 #NOTE: RESET PADDED ENTRIES TO ZERO.  THERE'S GOT TO BE AN EASIER WAY TO JUST PAD WITH ZEROS THOUGH.
            event_info        = np.concatenate((rec_particle_info,rec_traj_info),axis=1)
            
            # Append event level data to batch level data
            batch_info  = np.concatenate((batch_info,event_info),axis=0) if batch_info is not None else np.array(event_info)#NOTE: CHECK THIS...
            batch_truth = np.concatenate((batch_truth,event_truth),axis=0) if batch_truth is not None else np.array(event_truth)#NOTE: CHECK THIS...
        
        # Write batch data to file
        fmt = ["%.3g" for i in range(np.shape(batch_info)[1])]
        with open(data_file_name, "ab" if batch_num>0 else "wb") as f:
            rec_particle_header = [rec_particle_keys[idx] for idx in rec_particle_entry_indices]
            rec_traj_header = [key+'_'+str(i) for key in [rec_traj_keys[idx] for idx in rec_traj_entry_indices] for i in range(max_linked_entries)]
            header = replacement_header+delimiter.join([*rec_particle_header, *rec_traj_header]) if batch_num==0 else "" #NOTE: NOT GENERALIZED
            np.savetxt(f, batch_info, header=header, delimiter=delimiter, fmt=fmt)
            
        # Write batch truth to file
        fmt = ["%.3g" for i in range(np.shape(batch_truth)[1])]
        with open(truth_file_name, "ab" if batch_num>0 else "wb") as f:
            header = replacement_header+delimiter.join([mc_lund_keys[idx] for idx in truth_entry_indices]) if batch_num==0 else "" #NOTE: NOT GENERALIZED
            np.savetxt(f, batch_truth, header=header, delimiter=delimiter, fmt=fmt)
            
    remove_replacement_header(data_file_name,replacement_header=replacement_header)
    remove_replacement_header(truth_file_name,replacement_header=replacement_header)

#------------------------------ MAIN ------------------------------#
if __name__=="__main__":
    if len(sys.argv)<=3:
        print("Usage: python3 ",os.path.abspath(sys.argv[0])," outpath_xyz outpath_ptp file1 file2 file3 ...")
        sys.exit(0)
    out_path1 = os.path.abspath(sys.argv[1])
    out_path2 = os.path.abspath(sys.argv[2])
    file_list = sys.argv[3:]

    main( #TODO: use argparse package to make this cleaner
        file_list = file_list,
        banks = [
            'REC::Particle',
            'MC::Lund',
            'REC::Traj',
        ],
        step  = 1000,
        max_linked_entries = 30,
        data_file_name  = out_path1,
        truth_file_name = out_path2,
        delimiter       = ",",
        rec_particle_entry_indices = [1,2,3,4,5,6,7,9], # (pid), px, py, pz, vx, vy, vz, vt (Only in data), (charge), beta, (chi2pid), (status)
        rec_traj_entry_indices     = [2,3,4,5,6,7,8,9,10], # (pindex), (index), detector, layer, x, y, z, cx, cy, cz, path
        truth_entry_indices        = [6,7,8,11,12,13], # px, py, pz, vx, vy, vz
    )