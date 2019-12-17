import subprocess

from common.dataset import Dataset
from common.trajectory import Trajectory
from os.path import join
from skimage.io import imread
import numpy as np


def estimate_pose_by_1NN(frame_id, known_ids, known_poses, id_to_image):
    metric = lambda x, y: np.mean((x - y) ** 2)
    distances = {
        known_id: metric(id_to_image[frame_id], id_to_image[known_id])
        for known_id in known_ids
    }
    best_id, _ = min(distances.items(), key=lambda x: x[1])
    return known_poses[best_id]


def estimate_trajectory(data_dir, out_dir):
    # TODO: fill trajectory here

    id_to_filename = Dataset.read_dict_of_lists(join(data_dir, 'rgb.txt'))
    known_poses = Dataset.read_dict_of_lists(join(data_dir, 'known_poses.txt'))
    id_to_image = {
        frame_id: imread(join(data_dir, filename))
        for frame_id, filename in id_to_filename.items()
    }
    known_ids = set(known_poses.keys())
    unknown_ids = set(id_to_filename.keys()).difference(known_ids)

    estimated_poses = {
        frame_id: estimate_pose_by_1NN(frame_id, known_ids, known_poses, id_to_image)
        for frame_id in unknown_ids
    }
    all_poses = {
        **known_poses,
        **estimated_poses
    }

    # Trajectory.to_matrix4()
    trajectory = {
        frame_id: Trajectory.to_matrix4(positionAndQuaternion)
        for frame_id, positionAndQuaternion in all_poses.items()
    }
    Trajectory.write(Dataset.get_result_poses_file(out_dir), trajectory)

    # You may run some C++ code here
    subprocess.run('./bundle_adjustment 1.0 0.1', shell=True, check=True)
