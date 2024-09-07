
import numpy as np
from dm_control.utils import rewards
from scipy.spatial.transform import Rotation as R
from pyquaternion import Quaternion

# Height of head above which stand reward is 1.
_STAND_HEIGHT = 1.65 # The ideal height of the head is 1.68 m.

MAX_FOOT_DIST = 0.650 # It does not make sense consider the stride here. They are two different things. 
MIN_FOOT_DIST = 0.375 # when the robot spawns the distance between the feet is 0.4 m. I added a margin of 0.025 m for small errors.

STRIDE = 0.5

STANCE_MARGIN = 0.02


def head_height(named):
    data_temp = named.data
        
    quat = data_temp.xquat["torso_link"].copy()
    offset = np.array([0, 0, 0.7])
    offset = Quaternion(quat).rotate(offset)
    head_pos = data_temp.xpos["torso_link"] + offset #np.array([0, 0, 0.7])

    return head_pos[2] 

def torso_upright(named):
    return named.data.xmat["torso_link", "zz"]

class WalkV5Easy():

    def __init__(self,robot):
        super().__init__()
        print("[DEBUG basic_locomotion_tasks]: WalkV5Easy.__init__()")
        self._stand_height = _STAND_HEIGHT
        self._move_speed_lower_bound = 0.30 #0.83 # 3 km/h # 2 km/h = 0.55 
        self._move_speed_upper_bound = 1.78 # 6.4 km/h #4 km/h = 1.11 m/s

        self.ideal_orientation = Quaternion(np.array([1, 0, 0, 0]))

        self.reset()
        self.robot = robot
        

    def reset(self) -> None:
        self.phase = 0
        self.initial_phase = True
        
        self.old_l_foot_pos_x = 0.039468 # x at the beginning
        self.old_r_foot_pos_x = 0.039468 # x at the beginning



    def get_reward(self, robot, action: np.ndarray,named) -> float:
        self.robot = robot
        self.named = named
        standing = rewards.tolerance(
            head_height(named),
            bounds=(self._stand_height, float("inf")),
            margin=self._stand_height/8,
            value_at_margin=0,
            sigmoid="linear",
        )
        upright = rewards.tolerance(
            torso_upright(named),
            bounds=(0.9, float("inf")),
            sigmoid="linear",
            margin=1.9,
            value_at_margin=0,
        )
        stand_reward = standing * upright


        ctrl_ranges = self.robot.model.actuator_ctrlrange # robot.get_ctrl_ranges()
        actuator_forces = np.abs(robot.data.actuator_force) # The ctrl range is symmetric, so I can take the absolute value.
        actuator_forces = actuator_forces/ctrl_ranges[:, 1] # I divide by the maximum value of the control range.
        actuator_forces = np.mean(actuator_forces)
        control_reward = 1 - actuator_forces # I want to penalize the control signal. The reward is 1 minus the mean of the normalized control signal.
        
        velocity_x = named.data.sensordata["pelvis_subtreelinvel"][0] # robot.robot_velocity()[0] # I take only the x component of the velocity.
        position = robot.data.qpos[0:3] # I take only the y component of the position.

        position_y = position[1]

        # Smooth start: the robot start from 0. To avoid high acceleration at the beginning, I want to reward the robot for moving slowly at the beginning.
        if self.robot.data.time < 0.1:
            # speed_lower_bound must start from 0 and reach self._move_speed_lower_bound in 0.4 seconds
            speed_lower_bound = self._move_speed_lower_bound*self.robot.data.time/ 0.1
            #print("speed_lower_bound:", speed_lower_bound)
        else:
            speed_lower_bound = self._move_speed_lower_bound
        
        move = rewards.tolerance(
            velocity_x, 
            bounds=(speed_lower_bound, self._move_speed_upper_bound),
            margin=speed_lower_bound/2, #
            value_at_margin=0.1,
            sigmoid="gaussian",
        ) 

        centered_reward = rewards.tolerance(
            position_y,
            bounds=(-0.3, 0.3),
            margin=0.3,
            value_at_margin=0.1,
            sigmoid="linear",
        )
        
        orientation_quat = robot.data.qpos[3:7] # robot.robot_orientation()
        rotation = R.from_quat(orientation_quat.tolist(), scalar_first=True)
        angles = rotation.as_euler('zyx', degrees=True)
        angle_x = angles[0]
        stay_inline_reward = rewards.tolerance(
            angle_x,
            bounds=(-35, 35),
            margin=15,
            value_at_margin=0.1,
            sigmoid="linear",
        )

        
        phase_reward = self.get_reward_by_phase()

        print("stay_inline_reward:", stay_inline_reward, "centered_reward:", centered_reward, "move:", move, "phase_reward:", phase_reward, "control_reward:", control_reward, "stand_reward:", stand_reward)
        move = (move*centered_reward + move*stay_inline_reward)/2
        #reward = foot_reward*stand_reward*(2*move + 2*phase_reward + control_reward)/5
        reward = stand_reward*(2*move + 2*phase_reward + 2*control_reward)/6
        
        return reward, {
            "stand_reward": stand_reward,
            "small_control": control_reward,
            "move": move,
            "standing": standing,
            "upright": upright,
        }

    def get_reward_by_phase(self):
        l_foot_pos_x=self.named.data.xpos["left_ankle_link"][0]
        r_foot_pos_x=self.named.data.xpos["right_ankle_link"][0]

        #print("l_foot:", l_foot_pos_x, "r_foot:", r_foot_pos_x, "old_l:", self.old_l_foot_pos_x, "old_r:", self.old_r_foot_pos_x, "phase:", self.phase)

        if self.phase == 0: # right foot is swinging and left foot is stancing
            
            reward_l = rewards.tolerance(
                l_foot_pos_x,
                bounds=(self.old_l_foot_pos_x, self.old_l_foot_pos_x),
                margin = STANCE_MARGIN,
                value_at_margin = 0,
                sigmoid="linear",
            )

            if r_foot_pos_x > self.old_r_foot_pos_x:
                reward_r = 1
                self.old_r_foot_pos_x = r_foot_pos_x # I update the old foot position only if the foot is moving forward.
            else:
                reward_r = 0

            self.old_l_foot_pos_x = l_foot_pos_x # left foot must be stancing in this phase

            # If it is the first step STRIDE is only half of the normal value
            if self.initial_phase:
                if r_foot_pos_x > l_foot_pos_x + STRIDE/2:
                    self.initial_phase = False
                    self.phase = 1
            else:
                if r_foot_pos_x > l_foot_pos_x + STRIDE:
                    self.phase = 1 

        elif self.phase == 1: # left foot is swinging and right foot is stancing 
            reward_r = rewards.tolerance(
                r_foot_pos_x,
                bounds=(self.old_r_foot_pos_x, self.old_r_foot_pos_x),
                margin = STANCE_MARGIN,
                value_at_margin = 0,
                sigmoid="linear",
            )

            if l_foot_pos_x > self.old_l_foot_pos_x:
                reward_l = 1
                self.old_l_foot_pos_x = l_foot_pos_x # I update the old foot position only if the foot is moving forward.
            else:
                reward_l = 0

            self.old_r_foot_pos_x = r_foot_pos_x # right foot must be stancing in this phase

            if l_foot_pos_x > r_foot_pos_x + STRIDE:
                self.phase = 0
        
        
        else:
            raise ValueError("Phase not valid")
        
        return (reward_l+reward_r+reward_l*reward_r)/3