
# This file contains the TrajectoryInterpolator class, which is responsible for
# computing the desired acceleration at any given time based on a several different
# interpolation methods.

import numpy as np
from scipy.interpolate import CubicHermiteSpline

class TrajectoryInterpolator:

    def __init__(self, starting_qpos, starting_qvel , duration ,final_qpos, final_qvel, time_step):
        self.starting_qpos = starting_qpos
        self.starting_qvel = starting_qvel
        self.duration = duration
        self.final_qpos = final_qpos
        self.final_qvel = final_qvel
        self.time_step = time_step

        # Setup the interpolator
        self.setup()


    def setup(self):
        self.a0 = self.starting_qpos
        self.a1 = self.starting_qvel
        self.a2 = (3*(self.final_qpos - self.starting_qpos) - (2*self.starting_qvel + self.final_qvel)*self.duration) / (self.duration**2)
        self.a3 = (2*(self.starting_qpos - self.final_qpos) + (self.starting_qvel + self.final_qvel)*self.duration) / (self.duration**3)


    def get_acc(self,qpos, qvel, t):
        """Compute the desired position at time t"""
        t = round(t, 4)
        if t > self.duration:
            print("t", t, "shape", self.final_qvel.shape)
            return np.zeros_like(self.final_qvel)
        # if t == 0:
        #     print("t", t)

        if t == self.duration:
            print("t", t)
        return 6*self.a3*t + 2*self.a2
    
    def get_pos(self, t):
        """Compute the desired position at time t"""

        if t > self.duration:
            return self.final_qpos

        return self.a0 + self.a1*t + self.a2*t**2 + self.a3*t**3
        


if __name__ == "__main__":

    qpos_start = np.array([1,1])
    qpos_end = np.array([2,2])

    qvel_start = np.array([1,1])
    qvel_end = np.array([2,2])

    trajectory_interpolator = TrajectoryInterpolator(qpos_start, qvel_start, 1, qpos_end, qvel_end)

    print("positions test")
    print(trajectory_interpolator.get_pos(0.0))
    print(trajectory_interpolator.get_pos(0.5))
    print(trajectory_interpolator.get_pos(1.0))
    print(trajectory_interpolator.get_pos(1.5))

    print("accelerations test")
    print(trajectory_interpolator.get_acc(qpos_start, qvel_start, 0.0))
    print(trajectory_interpolator.get_acc(qpos_start, qvel_start, 0.5))
    print(trajectory_interpolator.get_acc(qpos_start, qvel_start, 1.0))
    print(trajectory_interpolator.get_acc(qpos_start, qvel_start, 1.5))
    
