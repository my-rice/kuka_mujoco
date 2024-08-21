
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


    def get_acc_by_derivative(self, qpos, qvel, t):
        """Compute the desired acceleration at time t"""

        pos_error = self.final_qpos - qpos
        vel_error = self.final_qvel - qvel

        # Use the derivative of the error to compute the desired acceleration
        desired_acc = pos_error/self.time_step

        return desired_acc


    def get_acc(self,qpos, qvel, t):
        """Compute the desired acceleration at time t"""

        # Use spline interpolation

        # Compute the desired acceleration at time t
        
        

        return self.spline(t, 2)