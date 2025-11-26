import yaml

class ValueProvider:
    def __init__(self, value_or_func):
        if callable(value_or_func):
            self.func = value_or_func
        else:
            self.func = lambda: value_or_func
    
    def get(self):
        return self.func()

class Config:
    """
    questa classe serve per assegnare i valori presenti nel file di configurazione yaml
    agli attributi della classe che chiama Config(file.yaml)
    """
    def __init__(self, filename: str):
        """
        The Config constructor takes as input a .yaml file and creates
        an attribute for each line with "name: value".

        Parameters
        -----------
        filename : str
            .yaml file used to set the attributes of the class  
        """
        with open(filename, "r") as f:
            cfg_dict = yaml.safe_load(f)

        for key, value in cfg_dict.items():
            setattr(self, key, ValueProvider(value))
            
            def make_setter(name):
                def setter(self, value: ValueProvider):
                    setattr(self, name, value)
                return setter
            
            setattr(self, f"set_{key}", make_setter(key)) #Closes over name! Assign a setter for each attribute