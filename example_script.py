import time

from XLO_sim.XLO_sim import XLO_sim
from XLO_sim import tools

n_repetitions = 10

start_time = time.perf_counter()

for i in range(n_repetitions):
    print(f"Repetition {i+1}/{n_repetitions}")
    X = XLO_sim("config/base/Cu-seed-SASE.yaml")

    seed_field = tools.Ocelot_SASE_seed_pstxy(X)
    X.configure(seed_field)
    X.run_3D()

elapsed_time = time.perf_counter() - start_time
print(f"Execution took {elapsed_time:.4f} seconds")