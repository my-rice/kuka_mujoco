import mujoco
import mediapy
import numpy as np
from logger import Logger

from trajectory_interpolator import TrajectoryInterpolator

import os
import imageio

PATH_TO_MODEL = "kuka_iiwa_14/scene.xml"
VIDEO_DIR = "results/"
DATA_DIR = "results/"


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
logger = Logger(DATA_DIR + "log.csv", ["time", "qpos", "qvel", "qacc", "ctrl"])
# Log the initial state
message = {
    "time": data.time,
    "qpos": data.qpos.copy(),
    "qvel": data.qvel.copy(),
    "qacc": data.qacc.copy(),
    "ctrl": data.ctrl.copy(),
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

trajectory_interpolator = TrajectoryInterpolator(qpos_start, np.zeros(7), duration, qpos_end, np.zeros(7))

#model.opt.disableflags = 1 << 4 # disable contact constraints


while data.time < duration:
    
    # compute desired acceleration
    t = data.time
    error = target_q - data.qpos

    
    prev_acc = data.qacc.copy()

    # inverse dynamics to compute required torque.
    # set qacc to the desired acceleration
    #target_acc = trajectory_interpolator.get_acc(data.qpos, data.qvel, t)
    target_acc = trajectory_interpolator.get_acc_by_derivative(data.qpos, data.qvel, t)
    data.qacc[:] = target_acc

    mujoco.mj_inverse(model, data)

    #print("data.solver_fwdinv", data.solver_fwdinv) # See if the inverse dynamics will be equal to the computed torque
    data.qacc[:] = prev_acc

    # set torque as control
    data.ctrl[:] = data.qfrc_inverse.copy()

    mujoco.mj_step(model, data)

    if len(frames) < data.time * framerate:
        renderer.update_scene(data, scene_option=dict())
        pixels = renderer.render()
        frames.append(pixels)
    
    message = {
        "time": data.time,
        "qpos": data.qpos.copy(),
        "qvel": data.qvel.copy(),
        "qacc": data.qacc.copy(),
        "ctrl": data.ctrl.copy(),
    }
    logger.log(message)


if renderer is not None:
    renderer.close()

imageio.mimsave(os.path.join(VIDEO_DIR, "video.mp4"), frames, fps=framerate)
logger.save()
#logger.plot_data()
