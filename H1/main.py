import mujoco
import mediapy
import numpy as np
from logger import Logger
import os
import imageio

from trajectory_interpolator import TrajectoryInterpolator

from dmc_wrapper import MjDataWrapper, MjModelWrapper
from dm_control.mujoco import index
from dm_control.mujoco.engine import NamedIndexStructs
from test_v5Easy import WalkV5Easy
from scipy.spatial.transform import Rotation as R

# README: run this file from the root of the repository: kuka_mujoco.

PATH_TO_MODEL = "./H1/h1/scene.xml"
VIDEO_DIR = "./results/h1/"
DATA_DIR = "./results/h1/"

# Create an enum to switch between the different solutions:
class Solution:
    CONTROL_CLIP = 1
    CONTROL_NOT_CLIP = 2
    Visualize = 3

solution = Solution.Visualize

# Instantiate the interpolator
#target_q = np.array([0.0108746505, 0.409904735, -0.04571992, 0.363424591, 0.91689547, 0.164697949, 0.0099925136, -1.08606159, -3.05513108, 2.39996658, -1.58373738, -1.53605197, 0.357645804, -3.9112009, 1.06593459, -0.612316674, -1.70175744, -0.158552603, 0.316076873, 1.62802804, 0.393499762, -0.880202047, -1.0811211, -0.900504928, -0.569864469, -1.42788048])
#target_vel = np.array([-3.34899157,56.3294191, -125.891773, 121.353618, 5.2636477, -199.343902, -63.5353335, -235.091203, 465.345048, -59.6334671, -341.345224, 72.1901469, -292.451727, 171.757122, -131.385152, -235.177784, 151.344121, 131.025596, 437.882499, 141.373359, -105.535527, -136.104178, 4.83413384, -102.166934, -127.759095]) 

def run(model, data, renderer, logger, traj_element, frames, framerate=30, named=None):
    
    # Getting the information from the trajectory.
    timestep = traj_element[0]
    target_q = traj_element[1:27]

    qpos_start = data.qpos.copy()
    qpos_end = target_q

    starting_vel = data.qvel.copy()
    target_vel = traj_element[27:52]
    #print("target_vel", target_vel)


    if solution != Solution.Visualize:
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

        trajectory_interpolator3 = TrajectoryInterpolator(starting_angles, starting_vel[3:6], duration, target_angles, target_vel[3:6], time_step=model.opt.timestep)

    t = 0
    while data.time <= timestep:
        
        t += model.opt.timestep
        if solution == Solution.Visualize:
            data.qpos = target_q
            data.qvel = target_vel
            mujoco.mj_forward(model, data)
            data.time += model.opt.timestep

            message = {
                "time": data.time,
                "left_foot_x": named.data.xpos["left_ankle_link"][0],
                "left_foot_y": named.data.xpos["left_ankle_link"][1],
                "left_foot_z": named.data.xpos["left_ankle_link"][2],
                "right_foot_x": named.data.xpos["right_ankle_link"][0],
                "right_foot_y": named.data.xpos["right_ankle_link"][1],
                "right_foot_z": named.data.xpos["right_ankle_link"][2],
                "distance": np.linalg.norm(named.data.xpos["left_ankle_link"] - named.data.xpos["right_ankle_link"]),
                "distance_x": named.data.xpos["left_ankle_link"][0] - named.data.xpos["right_ankle_link"][0]
            }


        else:
            

            prev_acc = data.qacc.copy()
            
            target_acc = np.zeros(25)
            target_acc[6:25] = trajectory_interpolator.get_acc(t)
            target_acc[0:3] = trajectory_interpolator2.get_acc(t)
            target_acc[3:6] = trajectory_interpolator3.get_acc(t)

            data.qacc[6:25] = target_acc[6:25]
            data.qacc[0:3] = target_acc[0:3]
            data.qacc[3:6] = target_acc[3:6]

            mujoco.mj_inverse(model, data)
            
            if solution == Solution.CONTROL_CLIP:
                # Solution 1. It is not good because the control will be clipped. If the control is clipped, the performance of the controller will be affected.
                # Here the robot has a free joint, so the net torque is not applied to the free joint if we use data.ctrl.
                # To solve the problem, the terms of the control that are applied to the free joint are copied to data.qfrc_applied.

                data.ctrl = data.qfrc_inverse.copy()[6:25]
                data.qfrc_applied[0:6] = data.qfrc_inverse[0:6]

                # Getting info about the performance. If any value in data.ctrl is greater of the maximum value, print a warning with the value of the control.
                
                ctrl = data.qfrc_inverse.copy()[6:25]
                for i in range(len(ctrl)):
                    if ctrl[i] > model.actuator_ctrlrange[i][1]:
                        print("Warning: control value is greater than the maximum value", ctrl[i], "max value", model.actuator_ctrlrange[i][1])
                    if ctrl[i] < model.actuator_ctrlrange[i][0]:
                        print("Warning: control value is smaller than the minimum value", ctrl[i], "min value", model.actuator_ctrlrange[i][0])
            
            elif solution == Solution.CONTROL_NOT_CLIP:
                # Solution 2. This works every time. The control is not clipped.
                data.qfrc_applied = data.qfrc_inverse.copy()

            data.qacc = prev_acc
            mujoco.mj_step(model, data)

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

        if len(frames) < data.time * framerate:
            renderer.update_scene(data, camera="top")
            pixels = renderer.render()
            frames.append(pixels)
        


if __name__ == "__main__":

    # Load the trajectory
    #traj = np.load("./H1/traj.npy")
    #traj = np.concatenate([np.array([0.400]), target_q, target_vel]).reshape(1, 52)
    
    traj = np.load("./H1/traj_straight_line.npy")
    print(traj.shape)

    # Load the model
    model = mujoco.MjModel.from_xml_path(PATH_TO_MODEL)
    data = mujoco.MjData(model)

    # named data structure
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
    
    # Raise the control range to eventually increase the performance of the controller.
    # value = 100000
    # model.actuator_ctrlrange = np.array([[-value, value]] * model.nu)

    framerate = 30  # (Hz)

    # Simulate and display video.
    frames = []
    mujoco.mj_resetData(model, data)  # Reset state and time.

    # Create a logger

    if solution == Solution.Visualize:
        header = ["time"]
        header += ["left_foot_x"]
        header += ["left_foot_y"]
        header += ["left_foot_z"]
        header += ["right_foot_x"]
        header += ["right_foot_y"]
        header += ["right_foot_z"]
        header += ["distance"]
        header += ["distance_x"]
    else:
        header = ["time"]
        header += [f"qpos_{i}" for i in range(model.nq)]
        header += [f"qvel_{i}" for i in range(model.nv)]
        header += [f"qacc_{i}" for i in range(model.nv)]
        header += [f"ctrl_{i}" for i in range(model.na)]

    logger = Logger(DATA_DIR + "log.csv", header)
    horizon = 5
    n_steps = traj.shape[0]//horizon

    n_steps = 1000

    # Reward
    reward_obj = WalkV5Easy(data)
    reward = 0
    class Robot():
        def __init__(self,model=None,data=None):
            self.model = model
            self.data = data
        def update(self,model=None,data=None):
            self.data = data
            self.model = model

    robot = Robot(model,data)
    # Run the simulation for n_steps trajectories elements
    for i in range(n_steps):

        traj_index = (i+1)*horizon -1
        
        run(model, data, renderer, logger, traj[traj_index],frames, framerate=framerate,named=named)
        # Get the reward
        robot.update(model,data)
        reward += reward_obj.get_reward(robot, None, named)[0]
        
        # Plot the qpos
        target_q = traj[traj_index][1:27]
        target_vel = traj[traj_index][27:52]
        if solution != Solution.Visualize:
            logger.plot_columns(DATA_DIR+ f"qpos_iter{i}.png", columns_names=[f"qpos_{i}" for i in range(model.nq)], references = [target_q[i] for i in range(model.nq)])
        
        # BUG. TODO: Solve the bugs in the plotting of other variables
        #logger.plot_columns(DATA_DIR+ f"ctrl_iter{i}.png", columns_names=[f"ctrl_{i}" for i in range(model.nv)], references = [0 for i in range(model.nv)])

        # print("position error", target_q-data.qpos)
        # print("velocity error", target_vel-data.qvel)
    # if renderer is not None: # BUG renderer.close() crashes the program
    #     renderer.close()

    logger.plot_columns(DATA_DIR+ f"feet.png", columns_names=["left_foot_x", "left_foot_y", "left_foot_z", "right_foot_x", "right_foot_y", "right_foot_z","distance","distance_x"], references = None)

    print("reward", reward)    

    # Save the video
    imageio.mimsave(os.path.join(VIDEO_DIR, "video.mp4"), frames, fps=framerate)
        