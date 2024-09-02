import mujoco
import mediapy
import numpy as np
from logger import Logger

from trajectory_interpolator import TrajectoryInterpolator

import os
import imageio

from scipy.spatial.transform import Rotation as R


PATH_TO_MODEL = "./H1/h1/scene.xml"
VIDEO_DIR = "./results/h1/"
DATA_DIR = "./results/h1/"

# README: run this file from the root of the repository: kuka_mujoco.

# Instantiate the interpolator
#target_q = np.array([0.0108746505, 0.409904735, -0.04571992, 0.363424591, 0.91689547, 0.164697949, 0.0099925136, -1.08606159, -3.05513108, 2.39996658, -1.58373738, -1.53605197, 0.357645804, -3.9112009, 1.06593459, -0.612316674, -1.70175744, -0.158552603, 0.316076873, 1.62802804, 0.393499762, -0.880202047, -1.0811211, -0.900504928, -0.569864469, -1.42788048])
#target_vel = np.array([-3.34899157,56.3294191, -125.891773, 121.353618, 5.2636477, -199.343902, -63.5353335, -235.091203, 465.345048, -59.6334671, -341.345224, 72.1901469, -292.451727, 171.757122, -131.385152, -235.177784, 151.344121, 131.025596, 437.882499, 141.373359, -105.535527, -136.104178, 4.83413384, -102.166934, -127.759095]) 

def run(model, data, renderer, logger, traj_element, frames, framerate=30):
    
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

    # Getting the information from the trajectory.
    timestep = traj_element[0]
    target_q = traj_element[1:27]

    qpos_start = data.qpos.copy()
    qpos_end = target_q

    starting_vel = data.qvel.copy()
    target_vel = traj_element[27:52]
    #print("target_vel", target_vel)

    # Interpolating the trajectory

    # Get the total duration of the trajectory
    duration = np.round(timestep - data.time, 3)
    
    # Instantiate the interpolator. The first will interpolate the position of the joints, the second the position of the free joint and the third the orientation of the free joint. These interpolators will be used to interpolate the acceleration of the joints.
    trajectory_interpolator = TrajectoryInterpolator(qpos_start[7:26], starting_vel[6:25], duration, qpos_end[7:26], target_vel[6:25], time_step=model.opt.timestep)
    trajectory_interpolator2 = TrajectoryInterpolator(qpos_start[0:3], starting_vel[0:3], duration, qpos_end[0:3], target_vel[0:3], time_step=model.opt.timestep)

    target_rotation = R.from_quat(target_q[3:7].tolist(), scalar_first=True)
    target_angles = target_rotation.as_euler('xyz', degrees=False)

    starting_rotation = R.from_quat(qpos_start[3:7].tolist(), scalar_first=True)
    starting_angles = starting_rotation.as_euler('xyz', degrees=False)

    # print("starting_angles", starting_angles)
    # print("target_angles", target_angles)
    trajectory_interpolator3 = TrajectoryInterpolator(starting_angles, starting_vel[3:6], duration, target_angles, target_vel[3:6], time_step=model.opt.timestep)

    # print("data.qpos", data.qpos)
    # print("data.ctrl", data.ctrl)
    # print("data.qacc", data.qacc)
    t = 0
    while data.time <= timestep:
        
        t += model.opt.timestep

        prev_acc = data.qacc.copy()
        

        target_acc = np.zeros(25)

        target_acc[6:25] = trajectory_interpolator.get_acc(t)
        target_acc[0:3] = trajectory_interpolator2.get_acc(t)
        target_acc[3:6] = trajectory_interpolator3.get_acc(t)


        data.qacc[6:25] = target_acc[6:25]
        data.qacc[0:3] = target_acc[0:3]
        data.qacc[3:6] = target_acc[3:6]

        mujoco.mj_inverse(model, data)
        
        # Solution 1. It is not good because the control will be clipped. If the control is clipped, the performance of the controller will be affected.
        # Here the robot has a free joint, so the net torque is not applied to the free joint if we use data.ctrl.
        # To solve the problem, the terms of the control that are applied to the free joint are copied to data.qfrc_applied.

        # data.ctrl = data.qfrc_inverse.copy()[6:25]
        # data.qfrc_applied[0:6] = data.qfrc_inverse[0:6]

        # Getting info about the performance. If any value in data.ctrl is greater of the maximum value, print a warning with the value of the control.
        
        # ctrl = data.qfrc_inverse.copy()[6:25]
        # for i in range(len(ctrl)):
        #     if ctrl[i] > model.actuator_ctrlrange[i][1]:
        #         print("Warning: control value is greater than the maximum value", ctrl[i], "max value", model.actuator_ctrlrange[i][1])
        #     if ctrl[i] < model.actuator_ctrlrange[i][0]:
        #         print("Warning: control value is smaller than the minimum value", ctrl[i], "min value", model.actuator_ctrlrange[i][0])


        # Solution 2. This works every time. The control is not clipped.
        data.qfrc_applied = data.qfrc_inverse.copy()

        data.qacc = prev_acc
        mujoco.mj_step(model, data)

        if len(frames) < data.time * framerate:
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
    traj = np.load("./H1/traj.npy")
    #traj = np.concatenate([np.array([0.400]), target_q, target_vel]).reshape(1, 52)
    print(traj.shape)

    # Load the model
    model = mujoco.MjModel.from_xml_path(PATH_TO_MODEL)
    data = mujoco.MjData(model)

    # Create a renderer
    renderer = mujoco.Renderer(model, 480, 640)
    renderer.update_scene(data, camera="top")
    pixels = renderer.render()
    
    # Raise the control range to eventually increase the performance of the controller.
    # value = 100000
    # model.actuator_ctrlrange = np.array([[-value, value]] * model.nu)

    framerate = 30  # (Hz)

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

    horizon = 200
    n_steps = traj.shape[0]//horizon

    # Run the simulation for n_steps trajectories elements
    for i in range(n_steps):

        traj_index = (i+1)*horizon -1
        
        run(model, data, renderer, logger, traj[traj_index],frames, framerate=framerate)

        # Plot the qpos
        target_q = traj[traj_index][1:27]
        target_vel = traj[traj_index][27:52]
        logger.plot_columns(DATA_DIR+ f"qpos_iter{i}.png", columns_names=[f"qpos_{i}" for i in range(model.nq)], references = [target_q[i] for i in range(model.nq)])
        
        # BUG. TODO: Solve the bugs in the plotting of other variables
        #logger.plot_columns(DATA_DIR+ f"ctrl_iter{i}.png", columns_names=[f"ctrl_{i}" for i in range(model.nv)], references = [0 for i in range(model.nv)])

        print("position error", target_q-data.qpos)
        print("velocity error", target_vel-data.qvel)
    # if renderer is not None: # BUG renderer.close() crashes the program
    #     renderer.close()

    # Save the video
    imageio.mimsave(os.path.join(VIDEO_DIR, "video.mp4"), frames, fps=framerate)
        