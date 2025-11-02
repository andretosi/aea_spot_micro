import yaml

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
            setattr(self, key, value)