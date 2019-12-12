import subprocess

from common.dataset import Dataset
from common.trajectory import Trajectory


def estimate_trajectory(data_dir, out_dir):
    # TODO: fill trajectory here
    trajectory = {}
    Trajectory.write(Dataset.get_result_poses_file(out_dir), trajectory)

    # You may run some C++ code here
    subprocess.run('./bundle_adjustment 1.0 0.1', shell=True, check=True)
