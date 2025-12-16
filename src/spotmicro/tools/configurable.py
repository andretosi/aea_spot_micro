from spotmicro.tools.config import Config
import inspect
from functools import wraps

def configurable(cls):
    """
    Decorator function that extends the behaviour of the decorated class by wrapping its init and adding a config attribute.
    It injects the logic to extract kwargs from the class constructor and their defaults in the init, to then pass them to the central registry to hold.
    
    :param cls: This class is any object that can be configured, and as such should be linked to the central registry
    """
    
    original_init = cls.__init__
    init_sig = inspect.signature(original_init) #extract consstructor signature so that we can copypaste it on our wrapper

    params = {name: param.default for name, param in init_sig.parameters.items() if name != "self"}
    if "config" not in params.keys():
        raise ValueError(f"Every configurable class must have a positional argument named config. ({cls})") 
    elif not isinstance(params["config"], Config):
        raise TypeError(f"config parameter must be of Config type, was given type: {type(params['config'])}")

    
    #extract kwargs from the constructor. Used to build overridden_params dict
    default_params = {name: param.default for name, param in params if param.default is not inspect.Parameter.empty}
    if "config" in default_params.keys():
        raise ValueError(f"Parameter config must be positional argument, was given as keywprd argument inside ({cls})")
    

    #<----- WRAPPER ----->
    @wraps(original_init)#used to preserve everything about the wrapped class and pass it to the wrapper
    def __init__(self, config: Config, *args, **kwargs):
        """
        This is a wrapper constructor that basically injects some config logic around the constructor of the given class
        """
        #These lines do not modify the class!!
        bound = init_sig.bind(self, *args, **kwargs) #Map positional and keyword arguments to the corect parameter. Returns a dict
        bound.apply_defaults()#Fill the above map with default values when none were provided
        
        #extract params and their runtime value
        overridden_params = {
            name : value for name, value in bound.arguments.items()
            if name in default_params and config.is_acceptable_type(value)
        }
        
        original_init(self, *args, **kwargs) #run the original constructor

        config.register(cls, self, overridden_params) #register the initialized instance to config
        self.config = config
    #<----- END WRAPPER ----->

    #<----- SAVE LOGIC ------>
    #TODO
    def save(self, path : str):
        """
        Save the parameters of **all** configurable objects to the given file

        :param path: path to the file to dump the config in.
        :type path: str
        """
        pass
    #<----- END SAVE   ------>
    #<----- LOAD LOGIC ------>
    #TODO
    def load(self, path: str):
        """
        Override the current config for this component with the parameters defined in another config file

        :param path: Description
        :type path: str
        """
        pass
    #<----- END LOAD   ------>


    #change the class constructor with our wrapper and add functions
    cls.__init__ = __init__
    cls.load = load
    cls.save = save

    return cls