import numpy as np
from spotmicro.agent.input import Input
from spotmicro.devices.random_controller import RandomController

class Controller():
    def __init__(self):
       self._device = None

    #TODO: add parameters
    @classmethod
    def from_randomController(cls):
        self = cls()
        self._device = RandomController()
        
        return self

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