import time

from XLO_sim.XLO_sim import XLO_sim
from XLO_sim import tools

n_repetitions = 1

start_time = time.perf_counter()

for i in range(n_repetitions):
    print(f"Repetition {i+1}/{n_repetitions}")
    X = XLO_sim("/Users/parkinpham/Programming/Physics/DESY_Internship/XLO_sim_related_notebook/config/Cu-seed_40.00uJ.yaml")

    seed_field = tools.Gaussian_pulse_aniso_seed(X)
    X.configure(seed_field)
    X.run_3D()

elapsed_time = time.perf_counter() - start_time
print(f"Execution took {elapsed_time:.4f} seconds")