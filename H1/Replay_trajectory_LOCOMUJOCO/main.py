import mujoco
import numpy as np

import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../')))

print("cwd", os.getcwd())
from logger import Logger

import imageio

from dmc_wrapper import MjDataWrapper, MjModelWrapper
from dm_control.mujoco import index
from dm_control.mujoco.engine import NamedIndexStructs


PATH_TO_MODEL = "H1/h1/scene.xml"
VIDEO_DIR = "./results/loco_mujoco_traj/"
DATA_DIR = "./results/loco_mujoco_traj/"


# README: run this file from the root of the repository: kuka_mujoco.


def run(model, data, renderer, logger, traj_element, frames, framerate=30, named=None):
    
    x_pos = traj_element[1]
    z_pos = traj_element[2]
    data.qpos[0:7] = np.array([x_pos,0,1.045-z_pos,1,0,0,0])
    data.qvel[0:6] = np.array([traj_element[26],0,0,0,0,0])
    
    data.qpos[7:26] = traj_element[7:26] # 0 timestep. 25 qpos + 25 qvel. First joint is at 1+6
    data.qvel[6:] = traj_element[32:51] #26+6

    #data.qpos = np.concatenate([np.array([traj_element[1],0,0.98,1,0,0,0]), traj_element[7:26]]) # 0 timestep. 25 qpos + 25 qvel
    #data.qvel = np.concatenate([np.array([traj_element[26],0,0,0,0,0]), traj_element[32:51]])
    mujoco.mj_forward(model, data)

    renderer.update_scene(data, camera="top")
    pixels = renderer.render()
    frames.append(pixels)
    
    message = {
        "time": data.time,
        **{
            f"qpos_{i}": data.qpos[i] for i in range(model.nq)
        },
        **{
            f"qvel_{i}": data.qvel[i] for i in range(model.nv)
        },
        **{
            f"qacc_{i}": data.qacc[i] for i in range(model.nv)
        },
        **{
            f"ctrl_{i}": data.ctrl[i] for i in range(model.na)
        },
    }

    logger.log(message)


if __name__ == "__main__":

    # Load the trajectory
    traj = np.load("./H1/Replay_trajectory_LOCOMUJOCO/samples.npy", allow_pickle=True)
    print("Shape:",traj.shape)

    # Load the model
    model = mujoco.MjModel.from_xml_path(PATH_TO_MODEL)
    data = mujoco.MjData(model)

    data_wrapper = MjDataWrapper(data)
    model_wrapper = MjModelWrapper(model)
    axis_indexers = index.make_axis_indexers(model_wrapper)
    named = NamedIndexStructs(
        model=index.struct_indexer(model_wrapper, "mjmodel", axis_indexers),
        data=index.struct_indexer(data_wrapper, "mjdata", axis_indexers),
    )

    # Create a renderer
    renderer = mujoco.Renderer(model, 480, 640)
    renderer.update_scene(data, camera="top")
    pixels = renderer.render()
    
    framerate = 1/0.01 #60  # (Hz)

    # Simulate and display video.
    frames = []
    mujoco.mj_resetData(model, data)  # Reset state and time.

    # Create a logger
    header = ["time"]
    header += [f"qpos_{i}" for i in range(model.nq)]
    header += [f"qvel_{i}" for i in range(model.nv)]
    header += [f"qacc_{i}" for i in range(model.nv)]
    header += [f"ctrl_{i}" for i in range(model.na)]

    logger = Logger(DATA_DIR + "log.csv", header)

    # Creating the message to log
    message = {
        "time": data.time,
        **{
            f"qpos_{i}": data.qpos[i] for i in range(model.nq)
        },
        **{
            f"qvel_{i}": data.qvel[i] for i in range(model.nv)
        },
        **{
            f"qacc_{i}": data.qacc[i] for i in range(model.nv)
        },
        **{
            f"ctrl_{i}": data.ctrl[i] for i in range(model.na)
        },
    }

    logger.log(message)


    horizon = 1
    n_steps = traj.shape[0]//horizon
    n_steps = 500
    # Run the simulation for n_steps trajectories elements
    for i in range(n_steps):

        traj_index = (i+1)*horizon -1
        
        run(model, data, renderer, logger, traj[traj_index],frames, framerate=framerate,named=named)

        # Plot the qpos
        # target_q = traj[traj_index][1:27]
        # target_vel = traj[traj_index][27:52]
        # logger.plot_columns(DATA_DIR+ f"qpos_iter{i}.png", columns_names=[f"qpos_{i}" for i in range(model.nq)], references = [target_q[i] for i in range(model.nq)])
        
        # TODO: Solve the bugs in the plotting of other variables
        #logger.plot_columns(DATA_DIR+ f"ctrl_iter{i}.png", columns_names=[f"ctrl_{i}" for i in range(model.nv)], references = [0 for i in range(model.nv)])

        #print("position error", target_q-data.qpos)
        #print("velocity error", target_vel-data.qvel)
    # if renderer is not None:
    #     renderer.close()
    print("Simulation finished")
    # Save the video
    imageio.mimsave(os.path.join(VIDEO_DIR, "video.mp4"), frames, fps=framerate)
        