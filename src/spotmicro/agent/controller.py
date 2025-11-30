import numpy as np
from spotmicro.agent.input import Input
from spotmicro.devices.device import Device

class Controller():
    def __init__(self, device: Device):
       self._device = device

    def update(self) -> None:
        #@TODO: probabily need to do more?
        self._device.update()

    @property
    def input(self) -> Input:
        i = self._device.read()
        if self._check_sanity(i):
            return i
            
    def _check_sanity(self, i: Input) -> bool:
        i_arr = i.as_array
        assert np.all((i_arr <= 1.0) & (i_arr >= -1.0)), "Input is not normalized"
        return True
