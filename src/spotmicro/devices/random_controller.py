#Assignee: Nico
from spotmicro.devices.device import Device
from spotmicro.agent.input import Input
from spotmicro.config import Config
import numpy as np
import random, os


class RandomController(Device):
    def __init__(self, config="config/RandomControllerConfig.yaml", **kwargs):
        
        self._input = Input()
        self._config = RandomControllerConfig(config, **kwargs)

        self.walk_state = WalkState(self)
        self.turn_state = TurnState(self)
        self.still_state = StillState(self)

        self._state = State(self)

    def update(self) -> None:
        """
        Update the internal state of the controller.
        May randomly change the input
        """
        next_state = self._state.update()
        if self._state is not next_state: #Works only because there are no self-loops
            self._state = next_state
            #print(self._state)
            self._state.enter()

    def read(self) -> Input:
        """
        Obtain the current input to give to the agent
        """
        return self._input

class State:
    """
    Serves as the initial state for this class, and the default one
    """
    def __init__(self, controller: RandomController):
        self.controller = controller
    
    def __str__(self):
        return "Base state"

    def update(self):
        states = [self.controller.still_state, self.controller.turn_state, self.controller.walk_state]
        probabilities = [self.controller._config.p_base2still, self.controller._config.p_base2turn, self.controller._config.p_base2walk] #ORDER MATTERS HERE
        return self._next_state(states, probabilities)

    def _next_state(self, states: list, probabilities: list):
        """
        Randomly chose a state from the given ones (list of States), assigning to each a certain probability (list of probabilities)
        State in position 0 will have a probabilities[0] probability of being returned, and so on

        Parameters
        ------
        states: list[State]
            An ordered list of the states it is possible to transition to from the present state
        probabilities: list[float]
            An ordered list of floats that represent the probability of transitioning to the associated state
        """
        return random.choices(states, probabilities, k=1)[0]
        

class TurnState(State):
    def __str__(self):
        return "Turning state"

    def enter(self):
        vx, vy, w = self._sample_command()
        self.controller._input.update(vx=vx, vy=vy, w=w)
        self.remaining_steps = int(np.random.normal(
            self.controller._config.w_steps_mean,
            self.controller._config.w_steps_var
        ))
    
    def update(self):
        self.remaining_steps -= 1
        if self.remaining_steps <= 0:
            states = [self.controller.still_state, self.controller.walk_state]
            probabilities = [self.controller._config.p_turn2still, self.controller._config.p_turn2walk]
            return self._next_state(states, probabilities)
        else:
            return self
    
    def _sample_command(self):
        w = np.clip(np.random.normal(self.controller._config.w_mean, self.controller._config.w_var), -1.0, 1.0)
        R = np.clip(np.random.normal(self.controller._config.w_radius_mean, self.controller._config.w_radius_var), -1.0, 1.0) #Should be normalized.. does not cause any trouble for now
        vx = w*R

        return vx, 0.0, w

class WalkState(State):
    def __str__(self):
        return "Walk state"

    def enter(self):
        vx, vy, w = self._sample_command()
        self.controller._input.update(vx=vx, vy=vy, w=w)
        self.remaining_steps = int(np.random.normal(
            self.controller._config.v_steps_mean,
            self.controller._config.v_steps_var
        ))

    def update(self):
        self.remaining_steps -= 1
        if self.remaining_steps <= 0:
            states = [self.controller.still_state, self.controller.turn_state]
            probabilities = [self.controller._config.p_walk2still, self.controller._config.p_walk2turn]
            return self._next_state(states, probabilities)
        return self
    
    def _sample_command(self):
        vx, vy = np.clip(tuple(np.random.normal(self.controller._config.v_mean, self.controller._config.v_var)), (-1.0, -1.0), (1.0, 1.0))
        return vx, vy, 0.0

class StillState(State):
    def __str__(self):
        return "Still state"

    def enter(self):
        self.controller._input.update(vx=0.0, vy=0.0, w=0.0)
        self.remaining_steps = int(np.random.normal(
            self.controller._config.s_steps_mean,
            self.controller._config.s_steps_var
        ))

    def update(self):
        self.remaining_steps -= 1
        if self.remaining_steps <= 0:
            states = [self.controller.walk_state, self.controller.turn_state]
            probabilities = [self.controller._config.p_still2walk, self.controller._config.p_still2turn]
            return self._next_state(states, probabilities)
        return self

class RandomControllerConfig(Config):
    """
    Handle all parameters for the RandomController.
    This class sets all default values and boundaries for all the parameters needed
    """
    def __init__(self, filename: str, **kwargs):
        if os.path.exists(filename):
            super().__init__(filename)

        #Set the defaults
        attributes = {
            "p_base2still": 0.3, 
            "p_base2walk": 0.7,
            "p_base2turn": 0.0,
            "p_still2walk": 1.0,
            "p_still2turn": 0.0,
            "p_walk2still": 0.2,
            "p_walk2turn": 0.8,
            "p_turn2still": 0.1,
            "p_turn2walk": 0.9,
            "v_mean": (0.23, 0.0), #Calibrated so that there is just a 11% chance of going in reverse 
            "v_var": (0.2, 0.02),
            "v_steps_mean": 300,
            "v_steps_var":37, #v_steps_mean/8
            "w_mean": 0.0,
            "w_var": 0.4,
            "w_radius_mean": 0.3,
            "w_radius_var": 0.1,
            "w_steps_mean": 100,
            "w_steps_var": 13, #As with v_steps_var
            "s_steps_mean": 50,
            "s_steps_var": 10,
        }
    
        for key, value in attributes.items():
            if not hasattr(self, key):
                self.set_property(key, value)
        
        #TODO? also ensure these checks when setting stuff. Authomatic way of handling this? A transition table for each node? meh
        print(self.p_base2still)
        print(type(self.p_base2still))
        if self.p_base2still + self.p_base2walk + self.p_base2turn != 1.0:
            raise ValueError("Sum of probabilities outgoing of base state must be exactly 1.0")
        if self.p_still2walk + self.p_still2turn != 1.0:
            raise ValueError("Sum of probabilities outgoing of still state must be exactly 1.0")
        if self.p_walk2still + self.p_walk2turn != 1.0:
            raise ValueError("Sum of probabilities outgoing of walk state must be exaclty 1.0")
        if self.p_turn2still + self.p_turn2walk != 1.0:
            raise ValueError("Sum of probabilities outgoing of turn state must be exaclty 1.0")
