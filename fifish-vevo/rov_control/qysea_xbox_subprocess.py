import sys
import time
import numpy as np
from qysea_xbox import QyseaXbox
from multiprocessing.connection import Listener
import json
from pynput import keyboard
import threading

slices = [0, 1, 2, 3, 4, 5, 6] # non-zero action indices

def action_to_command(prediction):
    command = np.zeros(15)
    for i in range(4):
        command[i] = prediction[i]
    # command[4] = prediction[4]
    if prediction[5] > 0.40:
        command[5] = 1.
    else:
        command[5] = 0.
    if prediction[6] > 0.40:
        command[6] = 1.
    else:
        command[6] = 0.

    command[11] = prediction[11]
    command[12] = prediction[12]
    command = command.astype(np.float64)
    return command

class ControllerApp:
    def __init__(self, address):
        self.qysea_xbox = QyseaXbox(fps=30)
        self.listener = Listener(address)
        self.mode = "manual"
        self.stopped = False
        self.latest_command = None

    def switch_mode(self, key):
        if key == keyboard.Key.esc:
            self.stopped = True
            return False
        try:
            if key.char == 'm':
                self.mode = "manual"
                print("Switched to manual mode")
            elif key.char == 'a':
                self.mode = "automatic"
                print("Switched to automatic mode")
            elif key.char == 'r':
                print('Switching mode...')
                self.mode = "manual"
                time.sleep(0.1)
                self.mode = "automatic"
        except AttributeError:
            pass

    def manual_control(self):
        while not self.stopped:
            if self.mode == "manual":
                action_copy = self.qysea_xbox.latest_action.copy()
                self.latest_command = action_copy
                for i in range(5):
                    self.latest_command[i] *= 2.0

                # print(self.latest_command[:7])
                self.qysea_xbox.send_action(self.latest_command)
            time.sleep(0.01)

    def automatic_control(self):
        while not self.stopped:
            conn = self.listener.accept()
            while True:
                try:
                    message = conn.recv()
                    if message == 'close':
                        break
                    elif message == 'ping':
                        continue
                    elif 'human' in message:
                        self.mode = "manual"
                    if self.mode == "automatic":
                        prediction = np.array(json.loads(message))
                        command_manual = self.qysea_xbox.latest_action.copy()
                        for i in range(5):
                            prediction[i] += command_manual[i]
                        command = action_to_command(prediction)
                        self.latest_command = command
                        # print(self.latest_command[:7])
                        self.qysea_xbox.send_action(self.latest_command)
                    else:
                        continue
                except EOFError:
                    break
                time.sleep(0.01)
            conn.close()
        

    def run(self):
        threading.Thread(target=self.qysea_xbox.load_latest_action, daemon=True).start()
        threading.Thread(target=self.manual_control, daemon=True).start()
        threading.Thread(target=self.automatic_control, daemon=True).start()
        with keyboard.Listener(on_press=self.switch_mode) as listener:
            listener.join()

if __name__ == '__main__':
    address = ('localhost', 8080)
    if len(sys.argv) > 1:
        address = tuple(sys.argv[1].strip('()').replace("'", "").split(','))
        address = (address[0].strip(), int(address[1].strip()))
    app = ControllerApp(address)
    app.run()
