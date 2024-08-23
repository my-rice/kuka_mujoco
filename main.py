import mujoco
import mediapy
import numpy as np
from logger import Logger

from trajectory_interpolator import TrajectoryInterpolator

import os
import imageio
import copy

from mujoco_mpc import agent as agent_lib

PATH_TO_MODEL = "kuka_iiwa_14/scene.xml"
VIDEO_DIR = "results/"
DATA_DIR = "results/"


def run():
    # Load the model
    model = mujoco.MjModel.from_xml_path(PATH_TO_MODEL)
    data = mujoco.MjData(model)

    renderer = mujoco.Renderer(model, 480, 640)


    duration = 5  # (seconds)
    framerate = 30  # (Hz)
    n_steps = int(np.ceil(duration * framerate)) + 1

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
    # Log the initial state

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


    # Instantiate the interpolator
    target_q = np.array(
        [
            0.28932849,
            0.92062463,
            -0.9442432,
            -1.02045466,
            1.2824041,
            -0.40766866,
            -1.41391154,
        ]
    )

    qpos_start = data.qpos.copy()
    qpos_end = target_q

    target_vel = np.zeros(7)
    trajectory_interpolator = TrajectoryInterpolator(qpos_start, np.zeros(7), 5, qpos_end, target_vel, time_step=model.opt.timestep)

    #model.opt.disableflags = 1 << 4 # disable contact constraints



    while data.time < duration:
        
        # compute desired acceleration
        t = data.time
        prev_acc = data.qacc.copy()
        
        # q_step_t = trajectory_interpolator.get_acc(data.qpos, data.qvel, t)
        # vel = np.zeros(7)
        # mujoco.mj_differentiatePos(model, vel, model.opt.timestep ,data.qpos, q_step_t)
        # target_acc = np.zeros(7)
        # mujoco.mj_differentiatePos(model, target_acc,model.opt.timestep,data.qvel, vel)
                
        target_acc = trajectory_interpolator.get_acc(data.qpos, data.qvel, t)


        data.qacc[:] = target_acc
        mujoco.mj_inverse(model, data)
        # set torque as control
        # print ctrl range
        print("ctrl range", model.actuator_ctrlrange)

        data.ctrl[:] = data.qfrc_inverse.copy()
        #print("data.solver_fwdinv", data.solver_fwdinv) # See if the inverse dynamics will be equal to the computed torque
        data.qacc[:] = prev_acc
        mujoco.mj_step(model, data)



        if len(frames) < data.time * framerate:
            renderer.update_scene(data, scene_option=dict())
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


    if renderer is not None:
        renderer.close()

    imageio.mimsave(os.path.join(VIDEO_DIR, "video.mp4"), frames, fps=framerate)
    #logger.save()
    logger.plot_columns("results/qpos.png", columns_names=[f"qpos_{i}" for i in range(model.nq)], references = [target_q[i] for i in range(model.nq)])



if __name__ == "__main__":
    run()