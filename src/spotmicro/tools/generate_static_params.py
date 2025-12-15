"""
This scripts generates a python file containing a dataclass holding all the parameters of all the configurable classes imported.
This way, these classes can have static hints in the IDE 
"""

from spotmicro.tools.configurable import configurable, CONFIGURABLE_REGISTRY
from spotmicro.tools.config import Config

from spotmicro.devices.device import Device
from spotmicro.devices.joystick_device import Joystick
from spotmicro.devices.keyboard_device import Keyboard
from spotmicro.devices.random_controller import RandomController
from spotmicro.devices.smooth_random_controller import SmoothRandomController

from spotmicro.env.spotmicro_env import SpotmicroEnv 
from spotmicro.env.terrain import Terrain

from spotmicro.agent.agent import Agent 
from spotmicro.agent.agent import Controller
from spotmicro.agent.input import Input

def generate_dataclass_for(cls):
    #main roadmap could be to re-create the config, then dump it to file as it is
    print("Prova")

for cls in CONFIGURABLE_REGISTRY:
    print(f"Found: {cls.__name__}")
    generate_dataclass_for(cls)
