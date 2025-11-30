import yaml

class ValueProvider:
    """
    Helper class that provides a unified interface 
    """
    def __init__(self, value_or_func):
        if callable(value_or_func):
            self.func = value_or_func
        else:
            self.func = lambda: value_or_func
    
    def get(self):
        v = self.func()
        return v

    def set(self, value_or_func):
        if callable(value_or_func):
            self.func = value_or_func
        else:
            self.func = lambda: value_or_func

class Config:
    """
    questa classe serve per assegnare i valori presenti nel file di configurazione yaml
    agli attributi della classe che chiama Config(file.yaml)
    """
    def __init__(self, filename: str):
        """
        The Config constructor takes as input a .yaml file and creates
        an attribute for each line with "_name: value".
        Then, generates a getter (callable with name) and a setter (callable by assigning to name)

        Parameters
        -----------
        filename : str
            .yaml file used to set the attributes of the class  
        """
        with open(filename, "r") as f:
            cfg_dict = yaml.safe_load(f)

        for key, value in cfg_dict.items():
            self.set_property(key, value)
            
    
    def set_property(self, key: str, value):
        # Instance-level storage
        setattr(self, f"_{key}", ValueProvider(value))

        # Property factory
        def make_property(name):
            def getter(self):
                return getattr(self, f"_{name}").get()

            def setter(self, value, lower_bound=None, upper_bound=None):
                getattr(self, f"_{name}").set(value, lb=lower_bound, ub=upper_bound)

            return property(getter, setter)

        # Assign property to the *class*, not the instance
        if not hasattr(self.__class__, key):
            setattr(self.__class__, key, make_property(key))