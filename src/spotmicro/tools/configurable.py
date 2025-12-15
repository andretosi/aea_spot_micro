from spotmicro.tools.config import Config
import inspect
from functools import wraps

CONFIGURABLE_REGISTRY = []

def configurable(cls):
    """
    Decorator function that extends the behaviour of the decorated class by wrapping its init and adding a config attribute.
    It injects the logic to extract kwargs from the class constructor and their defaults in the init, to then pass them to the central registry to hold.
    
    :param cls: This class is any object that can be configured, and as such should be linked to the central registry
    """
    
    CONFIGURABLE_REGISTRY.append(cls) #Register the class, to eventually generate the parametric dataclass

    original_init = cls.__init__ #store the original init somewhere before modifying it with config logic
    sig = inspect.signature(original_init) #extract its signature so that we can copypaste it on our wrapper
    
    #extract kwargs from the constructor. Used to build overridden_params dict
    default_params = {name: param.default for name, param in sig.parameters.items() 
                if name != "self" and param.default is not inspect.Parameter.empty}

    #<----- WRAPPER ----->
    @wraps(original_init)#used to preserve everything about the wrapped class and pass it to the wrapper. Everything, but the signature
    def __init__(self, *args, config : Config = None, **kwargs):
        """
        This is a wrapper constructor that basically injects some config logic around the constructor of the given class
        """
        #These lines do not modify the class!!
        bound = sig.bind(self, *args, **kwargs) #Map positional and keyword arguments to the corect parameter. Returns a dict
        bound.apply_defaults()#Fill the above map with default values when none were provided
        
        #extract params and their runtime value
        overridden_params = {
            name : value for name, value in bound.arguments.items()
            if name in default_params and config.is_acceptable_type(value)
        }
        
        original_init(self, *args, **kwargs) #run the original constructor

        if config is None:
            config = Config() #Generate an empty config to fill with default parameters and overrides 
        
        config.register(cls, self, overridden_params) #register the initialized instance to config
        self.config = config
    #<----- END WRAPPER ----->

    #change the class constructor with our wrapper
    cls.__init__ = __init__

    return cls