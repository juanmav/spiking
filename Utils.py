from mpi4py import MPI
import time


def get_simulation_prefix(HYPER_COLUMNS, simulation_time):
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()

    if rank == 0:
        simulation_prefix = time.strftime("%Y%m%d-%H-%M-%S") + f'-{HYPER_COLUMNS}-{simulation_time}'
    else:
        simulation_prefix = None

    simulation_prefix = comm.bcast(simulation_prefix, root=0)
    return simulation_prefix

