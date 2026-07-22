import multiprocessing as mp
from itertools import repeat
import time
import sys
import numpy as np
from XLO_sim import *
import Archiver as XLO_Archiver
from mpi4py import MPI

def single_realization(rep_idx, sim_config):
    """
    Calculate a single realization of XLO_sim.

    Parameters
    ----------
    rep_idx : int
        Index of the realization.

    sim_config : str
        Name of the configuration file for XLO_sim.

    Returns
    -------

    """
    X = XLO_sim(sim_config)
    X.configure()
    X.run_3D()
    return X


def mp_handler(pool, rep_idxs, sim_config):
    """
    Calculates multiple realizations of XLO_sim in a parallel way.

    Parameters
    ----------
    pool : mp.Pool
        Pool of processes.

    rep_idxs : list or ndarray
        Array of indices for the different realizations.

    sim_config : str
        Name of the configuration file for XLO_sim.

    Returns
    -------
    Xsim_array : list
        List of XLO_sim objects

    """
    Xsim_array = pool.starmap(single_realization, zip(rep_idxs, repeat(sim_config)))
    return Xsim_array

def calculate_cycles(Archiver):
    """
    Calculates the number of cycles for parallel execution considering the specified number of repetitions and number of available processes, and the number of realizations in the last cycle.

    Parameters
    ----------
    Archiver : XLO_Archiver
        XLO_Archiver object.

    Returns
    -------
    ncycles : int
        Number of cycles for parallel execution

    nlast : int
        Number of realizations in the last cycle.

    """
    nlast = Archiver.nrep % Archiver.nproc
    ncycles = Archiver.nrep // Archiver.nproc
    if nlast != 0:
        ncycles += 1
    return ncycles, nlast


def calculate_repetition_indices(i, ncycles, nlast):
    """
    Creates the array of indices for the given cycle of parallel execution.

    Parameters
    ----------
    i : int
        Cycle for which the indices are calculated

    ncycles : int
        Number of cycles for parallel execution

    nlast : int
        Number of realizations in the last cycle.

    Returns
    -------
    rep_inds : ndarray
        Array of indices for XLO_sim realizations in the specified cycle.

    """
    if nlast != 0 and i == ncycles-1:
        rep_idxs = np.arange(0, nlast)
    else:
        rep_idxs = np.arange(0, A.nproc)
    rep_idxs +=  i*A.nproc
    return rep_idxs


if __name__ == "__main__":
    comm=MPI.COMM_WORLD
    size = comm.Get_size()
    myid = comm.Get_rank()
    start = MPI.Wtime()
    if(myid==0) :
        print('Usage: python run_parallel.py sim_config archive_config data_directory')
        print('Starting  Maxwell-Bloch Simmulation MPI Parrelization over Initalization')
        print('Sim Config ', sys.argv[1] )
        print('Archive Config ', sys.argv[2] )
    out_file=sys.argv[3]+str(myid).zfill(5)+'.h5'
    A = XLO_Archiver.XLO_Archiver(sys.argv[1], sys.argv[2], out_file)
    A.setup_arrays()
    if(A.nproc !=size) :
        print('Nprocs must equal number of MPI Ranks')
        comm.abort()
    # MPI version
    ncycles, nlast = calculate_cycles(A)

    for i in range(ncycles):
        rep_idxs =  calculate_repetition_indices(i, ncycles, nlast)
        #print(f"\nStarting repetitions {', '.join(map(str, rep_idxs.tolist()))}...\n")

        #Xsim_array = mp_handler(pool, rep_idxs, A.sim_config_name)
        Xsim=single_realization(i, A.sim_config_name)
        A.update_arrays_for_single_realization(Xsim, i)
        del Xsim

    A.save_data()
    comm.Barrier()
    end = MPI.Wtime()
    exec_time = end - start
    if(myid==0) :
        print(f"Execution time: {exec_time} seconds")
                

