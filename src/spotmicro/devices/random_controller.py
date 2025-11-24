from spotmicro.devices.device import Device
from spotmicro.agent.input import Input


class RandomController(Device):
    def __init__(self):
        self._input = Input()

    def update(self) -> None:
        """
        Update the internal state of the controller.
        May randomly change the input
        """
        pass

    def read(self) -> Input:
        """
        Obtain the current input to give to the agent
        """
        return self._input