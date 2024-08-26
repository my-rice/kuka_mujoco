import numpy as np
import mujoco


class PositionController():

    def __init__(self):

        self.kp = np.array([200, 200, 200, 300, 40, 200, 200, 200, 300, 40, 300, 100, 100, 100, 100, 100, 100, 100, 100])
        self.kd = np.array([5, 5, 5, 6, 2, 5, 5, 5, 6, 2, 6, 2, 2, 2, 2, 2, 2, 2, 2])

    def control_step(self,model, data, desired_q_pos, desired_q_vel) -> np.array:
        
        # compute controller data
        mass_matrix = np.ndarray(shape=(model.nv, model.nv), dtype=np.float64, order="C")
        mujoco.mj_fullM(model, mass_matrix, data.qM)
        mass_matrix = np.reshape(mass_matrix, (len(data.qvel), len(data.qvel)))
        self.qvel_index = [i for i in range(6,25)]
        self.mass_matrix = mass_matrix[self.qvel_index, :][:, self.qvel_index]
        #print(self.mass_matrix)
        #print(self.mass_matrix.shape)
        self.qpos_index = [i for i in range(7,26)]
        self.joint_pos = np.array(data.qpos[self.qpos_index])
        self.joint_vel = np.array(data.qvel[self.qvel_index])

        position_error = desired_q_pos - self.joint_pos
        vel_pos_error = desired_q_vel-self.joint_vel
        desired_torque = np.multiply(np.array(position_error), np.array(self.kp)) + np.multiply(vel_pos_error, self.kd)

        # Get torque_compensation
        self.torque_compensation = data.qfrc_bias[self.qvel_index]

        # Return desired torques plus gravity compensations
        self.torques = np.dot(self.mass_matrix, desired_torque) + self.torque_compensation


        return self.torques

if __name__ == "__main__":

    PATH_TO_MODEL = "h1/scene.xml"

    traj = np.load("./traj.npy")
    print(traj.shape)

    # Load the model
    model = mujoco.MjModel.from_xml_path(PATH_TO_MODEL)
    data = mujoco.MjData(model)

    controller = PositionController()
    steps = 5
    for i in range(steps):
        target_qpos = traj[i][1:27]
        target_qvel = traj[i][27:52]
        assert len(target_qpos) == model.nq
        assert len(target_qvel) == model.nv
        #print("target_qpos", target_qpos)

        torques = controller.control_step(model,data,target_qpos[7:26],target_qvel[6:25])
        #print("torques",torques)

        data.ctrl[:] = torques
        mujoco.mj_step(model,data)

        print("Pos error:", target_qpos[:26] - data.qpos[:26])