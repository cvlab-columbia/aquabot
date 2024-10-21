import pygame
import numpy as np
import time
import random

class XboxController:
    def __init__(self):
        # Initialize pygame and joystick
        pygame.init()
        pygame.joystick.init()
        
        self.pygame = pygame

        # Initialize joystick
        self.joystick = pygame.joystick.Joystick(0)
        self.joystick.init()

        # Calibrate joystick
        self.neutral_positions = self.calibrate_joystick()

    def calibrate_joystick(self):
        print("Calibrating joystick. Please leave the joystick in the neutral position.")
        neutral_positions = []
        for _ in range(100):  # Capture 100 samples for calibration
            pygame.event.pump()
            neutral_positions.append([
                self.joystick.get_axis(0),
                self.joystick.get_axis(1),
                self.joystick.get_axis(2),
                self.joystick.get_axis(3),
                self.joystick.get_axis(4),
                self.joystick.get_axis(5)
            ])
            # time.sleep(0.01)  # Small delay to capture samples

        neutral_positions = np.array(neutral_positions)
        neutral_mean = neutral_positions.mean(axis=0)
        print("Calibration complete. Neutral positions:", neutral_mean)
        return neutral_mean

    def get_controller_input(self):
        pygame.event.pump()

        # Reset action array
        action = np.zeros(15)  # Assuming you have 10 actions to map

        '''
        0: left stick left/right    ->    left stick left/right
        1: left stick up/down       ->    left stick up/down
        2: right stick left/right   ->    right stick left/right
        3: right stick up/down      ->    right stick up/down
        4: left/right trigger       ->    left wheel left/right
        5: LB                       ->    right wheel left
        6: RB                       ->    right wheel right
        7: B                        ->    depth lock button unlock
        8: X                        ->    depth lock button lock
        9: Y                        ->    motor lock button unlock
        10: A                       ->    motor lock button lock
        11: left stick in           ->    light button 0
        12: right stick in          ->    light button 1
        13: start button            ->    reset ROV position
        14: back button             ->    reset global ROV position
        '''

        '''
        action statistics of data_0806
        #     mean    std     min     max
        0:  -0.001,  0.170, -1.017,  0.983
        1:   0.210,  0.527, -0.987,  1.013
        2:  -0.018,  0.342, -0.981,  1.019
        3:   0.560,  0.497, -0.977,  1.023
        4:  -0.109,  0.468, -2.000,  2.000
        5:   0.086,  0.280,  0.000,  1.000
        6:   0.011,  0.106,  0.000,  1.000
        7:   0.000,  0.000,  0.000,  0.000
        8:   0.000,  0.000,  0.000,  0.000
        9:   0.000,  0.000,  0.000,  0.000
        10:  0.000,  0.000,  0.000,  0.000
        11:  0.023,  0.149,  0.000,  1.000
        12:  0.000,  0.000,  0.000,  0.000
        13:  0.004,  0.060,  0.000,  1.000
        14:  0.000,  0.000,  0.000,  0.000
        '''

        # Map joystick axes to actions
        action[0] = 1 * (self.joystick.get_axis(0) - self.neutral_positions[0])  # left stick left/right
        action[1] = -1 * (self.joystick.get_axis(1) - self.neutral_positions[1])  # left stick up/down
        action[2] = 1 * (self.joystick.get_axis(3) - self.neutral_positions[3])  # right stick left/right
        action[3] = -1 * (self.joystick.get_axis(4) - self.neutral_positions[4])  # right stick up/down
        action[4] = -0.5 * (self.joystick.get_axis(2) - self.neutral_positions[2])  # left trigger
        action[4] += 0.5 * (self.joystick.get_axis(5) - self.neutral_positions[5])  # right trigger
        action[5] = 1 * self.joystick.get_button(4) # LB
        action[6] = 1 * self.joystick.get_button(5) # RB
        action[7] = 1 * self.joystick.get_button(1) # B
        action[8] = 1 * self.joystick.get_button(2) # X
        action[9] = 1 * self.joystick.get_button(3) # Y
        action[10] = 1 * self.joystick.get_button(0) # A
        action[11] = 1 * self.joystick.get_button(9) # left stick in
        action[12] = 1 * self.joystick.get_button(10) # right stick in
        action[13] = 1 * self.joystick.get_button(7) # start button
        action[14] = 1 * self.joystick.get_button(6) # back button
        
        return action

    def run(self):
        print("Running Xbox controller input capture. Press BACK button to exit.")
        while True:
            loop_start = time.perf_counter()
            action = self.get_controller_input()
            # action = 0
            # time.sleep(random.random() * 0.05)
            if action is None:
                print("Exiting...")
                break
            # print("Control values:", action)
            sleep_time = max(0, 0.1 - (time.perf_counter() - loop_start))
            time.sleep(sleep_time)  # Adjust the delay as needed
            print("current time: ", loop_start)

if __name__ == "__main__":
    controller = XboxController()
    # controller.run()

    start_time = time.perf_counter()
    for i in range(10000):
        controller.get_controller_input()
    
    print("elapsed time: ", time.perf_counter() - start_time)
