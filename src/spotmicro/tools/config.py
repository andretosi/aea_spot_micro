#TODO: implement this

class Config:
    #The constructor for the config class needs to generate the first layer of parameters with the ones coming from a config file
    def __init__(self, filepath: str = None):
        """
        Config class to gather parameters.
        
        :param self: Description
        :param filepath: path to the config.yaml file. If this parameter is not passed, then an empty config object is generated
        :type filepath: str
        """
        raise NotImplementedError("Config was not implemented yet")

    def save(self, dst_filepath: str):
        pass

    def load(self, src_filepath: str):
        pass

    def register(self, component_type, component_instance, params):
        """
        Add the parameters of an instance of a configurable class to the registry, in its namespace

        
        :param component_type: class of the component
        :param component_instance: specific instance of the component
        :param params: kwargs passed to the constructor of the provided instance of the component
        """
        
        if not isinstance(component_instance, component_type):
            raise ValueError("Mismatch between declared component type and type of the component instance provided")

        #If it doesn't already exist, create a dataclass for that component and make it a member of the config
        
        
        raise NotImplementedError("Not implemented yet")
    
    def is_acceptable_type(self, value) -> bool:
        """
        returnn True if the provided value is elegible for being saved in config.
        if the value is not a primitive typer, one with a clear, concise str representation or in general it doesn't make sense to dump it in a .yaml, then return False
        
        :param value: config parameter to filter
        """
        pass