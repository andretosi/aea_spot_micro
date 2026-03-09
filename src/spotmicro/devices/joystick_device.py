#Assignee mirko

from spotmicro.devices.device import Device
from spotmicro.agent.input import Input


class Joystick(Device):
    def __init__(self):
        import Gamepad
        import time

        # Gamepad settings
        gamepadType = Gamepad.PS3
        buttonExit = 'PS'
        vertical = 'LEFT-Y'
        horizzontal = 'LEFT-X'
        angoular = 'RIGHT-X'
        pollInterval = 0.2

        # Wait for a connection
        if not Gamepad.available():
            print('Please connect your gamepad...')
            while not Gamepad.available():
                time.sleep(1.0)
        gamepad = gamepadType()
        print('Gamepad connected')

        # Set some initial state
        global running
        running = True
        self.x = 0.0
        self.y = 0.0
        self.w = 0.0

        # Create some callback functions
        def exitButtonPressed():
            global running
            print('EXIT')
            running = False

        def verticalAxisMoved(position):
            global sideway
            self.x = -position #for some reason the guide suggest to invert linear velocities but not angoular

        def horizontalAxisMoved(position):
            global foward
            self.y = -position
        
        def angoularAxisMoved(position):
            global steering
            self.w = position


        # Start the background updating
        gamepad.startBackgroundUpdates()

        # Register the callback functions
        gamepad.addButtonPressedHandler(buttonExit, exitButtonPressed)
        gamepad.addAxisMovedHandler(vertical, verticalAxisMoved)
        gamepad.addAxisMovedHandler(horizontal, horizontalAxisMoved)
        gamepad.addAxisMovedHandler(angoular, angoularAxisMoved)

        # Keep running while joystick updates are handled by the callbacks
        try:
            while running and gamepad.isConnected():
                # Show the current speed and steering
                print('%+.1f %% self.x, %+.1f %% self.y, %+.1f %% self.w' % (x * 100, y * 100, w*100))


                # Sleep for our polling interval
                time.sleep(pollInterval)
            finally:
                # Ensure the background thread is always terminated when we are done
                gamepad.disconnect()
        

        raise NotImplementedError("Joystick device is not implemented yet")