import mujoco
import mediapy
import numpy as np
from logger import Logger

from trajectory_interpolator import TrajectoryInterpolator

import os
import imageio

from scipy.spatial.transform import Rotation as R


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
            1.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            1.0,
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

    target_vel = np.zeros(13)
    trajectory_interpolator = TrajectoryInterpolator(qpos_start[7:14], np.zeros(7), 5, qpos_end[7:14], target_vel[6:13], time_step=model.opt.timestep)
    trajectory_interpolator2 = TrajectoryInterpolator(qpos_start[0:3], np.zeros(3), 5, qpos_end[0:3], target_vel[0:3], time_step=model.opt.timestep)
    #model.opt.disableflags = 1 << 4 # disable contact constraints

    target_rotation = R.from_quat(target_q[3:7].tolist(), scalar_first=True)
    target_angles = target_rotation.as_euler('xyz', degrees=False)

    starting_rotation = R.from_quat(qpos_start[3:7].tolist(), scalar_first=True)
    starting_angles = starting_rotation.as_euler('xyz', degrees=False)

    print("starting_angles", starting_angles)
    print("target_angles", target_angles)
    trajectory_interpolator3 = TrajectoryInterpolator(starting_angles, np.zeros(3), 5, target_angles, np.zeros(3), time_step=model.opt.timestep)



    print("data.qpos", data.qpos)
    print("data.ctrl", data.ctrl)
    print("data.qacc", data.qacc)
    print("data.qfrc_inverse", data.qfrc_inverse)
    print("data.qfrc_applied", data.qfrc_applied)
    print("data.xfrc_applied", data.xfrc_applied)
    data.qfrc_applied = np.zeros(13)
    data.xfrc_applied = np.zeros((model.nbody, 6))

    vel = np.zeros(13)
    mujoco.mj_differentiatePos(model, vel, duration, data.qpos, target_q)
    print("vel", vel)

    while data.time < duration:
        
        
        t = data.time

        prev_acc = data.qacc.copy()
        
        target_acc = trajectory_interpolator.get_acc(data.qpos[7:14], data.qvel[6:13], t)


        data.qacc[6:13] = target_acc
        data.qacc[0:3] = trajectory_interpolator2.get_acc(data.qpos[0:3], data.qvel[0:3], t)
        data.qacc[3:6] = trajectory_interpolator3.get_acc(data.qpos[7:11], data.qvel[3:6], t)
        mujoco.mj_inverse(model, data)
        
        # set torque as control
        # print ctrl range
        #print("ctrl range", model.actuator_ctrlrange)
        #print("len(qfrc_inverse)", len(data.qfrc_inverse))
        data.ctrl = data.qfrc_inverse.copy()[6:13]
        data.qfrc_applied[0:6] = data.qfrc_inverse.copy()[0:6]
        #data.xfrc_applied = np.zeros((model.nbody, 6))

        #print("data.solver_fwdinv", data.solver_fwdinv) # See if the inverse dynamics will be equal to the computed torque
        # data.qacc[6:13] = prev_acc[6:13]
        # data.qacc[0:6] = np.zeros(6)
        data.qacc = prev_acc
        #print("data.qacc", data.qacc)
        mujoco.mj_step(model, data)

        
        # compute desired acceleration

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