# Library for the array, matrices, and high level mathematical calculations
import numpy as np

# OpenCV module
import cv2

# Math module
import math

# Time module
import time

# Matlab module
import matplotlib

# Analysis and design of feedback control systems
import control

# Class to represent the PID controller
class PID_controller:

    # Initialize the Proportional-Integral-Derivative Controller
    def __init__(self):

        # Previous action in torque
        self.prev_action = 0

        # Current value for integral
        self.integral = 0

        # Previous error
        self.previousError = 0

        # List to store all errors
        self.errorHistory = []

        # List to store all the integrals
        self.integralHistory = []

    # Function to reset the PID controller
    def reset_state(self):
        
        # Reset the error history
        self.errorHistory=[]

        # Reset the integral history
        self.integralHistory=[]

    # Function to return the average error
    def averageError(self):
        return sum(self.errorHistory)/len(self.errorHistory)

    # Function to calculate the error given the theta
    def errorCalculator(self,theta):
        
        # Compute the previous error as theta mod 2 * pi
        previous_error = (theta%(2*math.pi))-0

        # Check if the previous error is greater than PI
        if previous_error > math.pi:

            # Set the previous error to the previous error minus 2 * pi
            previous_error = previous_error - (2*math.pi)

        # Return the previous error        
        return previous_error

    # PID Controller
    def pidController(self,time_delta,error,previous_error,integral):

        # Average error
        averageError = self.averageError()

        # p = 2,0,0
        # d = 0,1,0
        # i = 0,0,0.1
        
        # p,d = 5,484,0
        # p,i = 2,0,0.1
        # i,d =  0,0.1,0.1

        # p,i,d = -0.1, -0.1, 0.1

        Kp = 2
        Kd = 0
        Ki = 0

        # settling time = bound

        # Calculate the proportional error
        proportional = (Kp*error)

        # Calculate the derivative error
        derivative = Kd*((error-previous_error)/time_delta)

        # Calculate the integral error
        integral += error * time_delta

        # Compute the force using the equation
        F = proportional + derivative + (Ki*integral)

        # Return the force calculated
        return F

    # Function to get the required force to be applied to the cart ~  image state is a (800, 400, 3) numpy image array
    def get_action(self,state,disturbance,image_state,random_controller=False):

        # boolean / int / float [-2.4,2.4] / float [-inf,inf] / float [-pi/2,pi/2] radians / float [-inf,inf] / int [0,1] 
        terminal,timestep,x,x_dot,theta,theta_dot,reward = state

        # If the random controller is selected
        if random_controller:

            # return a random force between -1 and 1 multiplied by a factor of 10
            return np.random.uniform(-1,1) * 10

        # If the PID controller is selected
        else:      
            
            # Compute the error using the theta angle
            error = self.errorCalculator(theta)

            # Add the error to the history of errors
            self.errorHistory.append(error)
            
            # Force calculated by the PID Controller
            force = self.pidController(timestep,error,self.previousError,self.integral)

            # Update the previous error
            self.previousError = self.errorHistory[-1]

            # Random Disturbance 1: Less than 0.1% of the time
            if disturbance and np.random.rand()>0.999:

                # Return a random force multiplied by a factor of 100 + the force calculated by the PID controller
                return np.random.rand()*100 + force
            
            # Return the force required to push the cart
            return force