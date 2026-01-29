import pyspacemouse
import time

success = pyspacemouse.open(dof_callback=pyspacemouse.print_state, button_callback=pyspacemouse.print_buttons)
if success:
    while 1:
        state = pyspacemouse.read()
        time.sleep(0.01)