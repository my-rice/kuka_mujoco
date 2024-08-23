
# This file contains the TrajectoryInterpolator class, which is responsible for
# computing the desired acceleration at any given time based on a several different
# interpolation methods.

import numpy as np
from scipy.interpolate import CubicHermiteSpline

class TrajectoryInterpolator:

    def __init__(self, starting_qpos, starting_qvel , duration ,final_qpos, final_qvel, time_step=0.002):
        self.starting_qpos = starting_qpos
        self.starting_qvel = starting_qvel
        self.duration = duration
        self.final_qpos = final_qpos
        self.final_qvel = final_qvel
        self.time_step = time_step

        # Create the spline
        times = np.array([0, self.duration])  # start and end times
        positions = np.array([self.starting_qpos, self.final_qpos])
        velocities = np.array([self.starting_qvel, self.final_qvel])

        # Create the cubic Hermite spline (Cubic Hermite is good for qpos and qvel)
        self.spline = CubicHermiteSpline(times, positions, velocities)

        # Setup the interpolator
        self.setup()

    # def get_acc_by_derivative(self, qpos, qvel, t):
    #     """Compute the desired acceleration at time t"""

    #     pos_error = self.final_qpos - qpos
    #     vel_error = self.final_qvel - qvel

    #     # Use the derivative of the error to compute the desired acceleration
    #     desired_acc = (pos_error/self.time_step - qvel)/self.time_step

    #     return desired_acc



    def setup(self):
        self.a0 = self.starting_qpos
        self.a1 = self.starting_qvel
        self.a2 = (3*(self.final_qpos - self.starting_qpos) - (2*self.starting_qvel + self.final_qvel)*self.duration) / (self.duration**2)
        self.a3 = (2*(self.starting_qpos - self.final_qpos) + (self.starting_qvel + self.final_qvel)*self.duration) / (self.duration**3)


    def get_acc(self,qpos, qvel, t):
        """Compute the desired position at time t"""

        if t > self.duration:
            return np.zeros(7)

        return 6*self.a3*t + 2*self.a2
    
    def get_pos(self, t):
        """Compute the desired position at time t"""

        if t > self.duration:
            return self.final_qpos

        return self.a0 + self.a1*t + self.a2*t**2 + self.a3*t**3
        


if __name__ == "__main__":

    qpos_start = np.array([0,0])
    qpos_end = np.array([1,1])

    qvel_start = np.array([0,0])
    qvel_end = np.array([0,0])

    trajectory_interpolator = TrajectoryInterpolator(qpos_start, qvel_start, 1, qpos_end, qvel_end)

    print(trajectory_interpolator.get_pos(0.5))
    print(trajectory_interpolator.get_pos(1.0))
    print(trajectory_interpolator.get_pos(1.5))

    print(trajectory_interpolator.get_acc(qpos_start, qvel_start, 0.5))
    print(trajectory_interpolator.get_acc(qpos_start, qvel_start, 1.0))
    print(trajectory_interpolator.get_acc(qpos_start, qvel_start, 1.5))
    
