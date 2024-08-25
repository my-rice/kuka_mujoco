import mujoco
import mujoco_viewer
import numpy as np

PATH_TO_MODEL = "kuka_iiwa_14/scene.xml"

# Load the model
model = mujoco.MjModel.from_xml_path(PATH_TO_MODEL)
data = mujoco.MjData(model)

# Create a viewer
viewer = mujoco_viewer.MujocoViewer(model, data)


# Set the target qpos
target_q = np.array(
  [
    0.25214491,
    0.78621591,
    -1.06877266,
    -0.96545645,
    0.65902108,
    -0.22038447,
    -0.88148042,
  ]
)

print("INITIAL qpos", data.qpos)

# data.xpos[1] = np.array([-1.5, -1.5, -1.5])
# data.xquat[1] = np.array([1, 0, 0, 0])


# Simulate and visualize
while True:
    
  #data.qpos[:len(target_q)] = target_q
  mujoco.mj_step(model, data)
  viewer.render()