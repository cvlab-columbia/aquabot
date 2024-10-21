import time
import sys
import threading
import numpy as np
from pynput import keyboard
from queue import Queue

sys.path.append('../')

from qysea.sdk.manage import QY_Rov_Manage
from qysea.sdk.manage.QY_RovController_Manage import QYRovControllerManage
from xbox import XboxController

def control_setting():
    print('====================')
    print('control setting:')
    qy_rov_pa_manage = QY_Rov_Manage.QYRovParameterManage()
    print('Set the controller operation mode to UAV_USA:')
    print(qy_rov_pa_manage.set_rov_controller_operation('UAV_USA'))
    print('Set the throttle curvature and limit: 0, 50')
    print(qy_rov_pa_manage.set_throttle_curvature_and_limit(0, 50))
    print(qy_rov_pa_manage.get_throttle_curvature_and_limit())
    print('Set the rotate curvature and limit: 0, 50')
    print(qy_rov_pa_manage.set_rotate_curvature_and_limit(0, 50))
    print(qy_rov_pa_manage.get_rotate_curvature_and_limit())
    print('Return to the original posture:')
    print(qy_rov_pa_manage.set_rov_posture_original())
    print('====================')
    return qy_rov_pa_manage

def connect_controller():
    print('====================')
    print('connect to controller:')
    qy_rov_controller_manage = QYRovControllerManage()
    print(qy_rov_controller_manage.set_remote_control_status("ON"))
    print('====================')
    return qy_rov_controller_manage

def initialize_joystick():
    print('Initialize and calibrating joystick:')
    return XboxController()

def load_trajectory(file_path):
    return np.load(file_path)

def control_loop(qy_rov_controller_manage, qy_rov_pa_manage, xbox, control_queue, state):
    recording = False
    trajectory = None

    def on_press(key):
        nonlocal recording, trajectory
        if key == keyboard.Key.space:
            if state['mode'] == 'manual':
                state['mode'] = 'replay'
                trajectory = load_trajectory("control_trajectory.npy")
                print("Started replaying the control action trajectory.")
            elif state['mode'] == 'replay':
                state['mode'] = 'manual'
                print("Switched back to manual control.")
                control_queue.put(None)  # Signal the end of the replay
            return False  # Stop listener

    listener = keyboard.Listener(on_press=on_press)
    listener.start()

    while True:
        if state['mode'] == 'manual':
            action = xbox.get_controller_input()
            start = time.time()
            qy_rov_controller_manage.set_left_joystick_left_right(int(action[0] * 500 + 1500))
            qy_rov_controller_manage.set_left_joystick_up_down(int(action[1] * 500 + 1500))
            qy_rov_controller_manage.set_right_joystick_left_right(int(action[2] * 500 + 1500))
            qy_rov_controller_manage.set_right_joystick_up_down(int(action[3] * 500 + 1500))
            qy_rov_controller_manage.set_left_wave_left_right(int(action[4] * 500 + 1500))
            if action[5] == 1:
                qy_rov_controller_manage.set_right_wave_left_right(1000)
            elif action[6] == 1:
                qy_rov_controller_manage.set_right_wave_left_right(2000)
            else:
                qy_rov_controller_manage.set_right_wave_left_right(1500)

            if action[7] == 1:
                qy_rov_controller_manage.set_keep_depth_button(1)
            elif action[8] == 1:
                qy_rov_controller_manage.set_keep_depth_button(0)

            if action[9] == 1:
                qy_rov_controller_manage.set_rc_lock_button(0)
            elif action[10] == 1:
                qy_rov_controller_manage.set_rc_lock_button(1)

            if action[11] == 1:
                qy_rov_controller_manage.set_right_switch(0)
            elif action[12] == 1:
                qy_rov_controller_manage.set_right_switch(1)

            if action[13] == 1:
                qy_rov_pa_manage.set_rov_posture_original()

            end = time.time()
            # print('Time elapsed:', end - start)

            time.sleep(0.05)  # 20 Hz control loop

        elif state['mode'] == 'replay' and trajectory is not None:
            for action in trajectory:
                qy_rov_controller_manage.set_left_joystick_left_right(int(action[0] * 500 + 1500))
                qy_rov_controller_manage.set_left_joystick_up_down(int(action[1] * 500 + 1500))
                qy_rov_controller_manage.set_right_joystick_left_right(int(action[2] * 500 + 1500))
                qy_rov_controller_manage.set_right_joystick_up_down(int(action[3] * 500 + 1500))
                qy_rov_controller_manage.set_left_wave_left_right(int(action[4] * 500 + 1500))
                if action[5] == 1:
                    qy_rov_controller_manage.set_right_wave_left_right(1000)
                elif action[6] == 1:
                    qy_rov_controller_manage.set_right_wave_left_right(2000)
                else:
                    qy_rov_controller_manage.set_right_wave_left_right(1500)

                if action[7] == 1:
                    qy_rov_controller_manage.set_keep_depth_button(1)
                elif action[8] == 1:
                    qy_rov_controller_manage.set_keep_depth_button(0)

                if action[9] == 1:
                    qy_rov_controller_manage.set_rc_lock_button(0)
                elif action[10] == 1:
                    qy_rov_controller_manage.set_rc_lock_button(1)

                if action[11] == 1:
                    qy_rov_controller_manage.set_right_switch(0)
                elif action[12] == 1:
                    qy_rov_controller_manage.set_right_switch(1)

                if action[13] == 1:
                    qy_rov_pa_manage.set_rov_posture_original()

                time.sleep(0.05)  # Maintain 20 Hz control loop

            state['mode'] = 'manual'
            print("Finished replaying the control action trajectory.")

def main():
    qy_rov_pa_manage = control_setting()
    qy_rov_controller_manage = connect_controller()
    xbox = initialize_joystick()

    control_queue = Queue()
    state = {'mode': 'manual'}

    control_thread = threading.Thread(target=control_loop, args=(qy_rov_controller_manage, qy_rov_pa_manage, xbox, control_queue, state))
    control_thread.start()

    control_thread.join()

    print(qy_rov_controller_manage.set_remote_control_status("OFF"))

if __name__ == '__main__':
    main()
