import yaml

class Config:
    def __init__(self, filename: str):
        with open(filename, "r") as f:
            cfg_dict = yaml.safe_load(f)

        for key, value in cfg_dict.items():
            setattr(self, key, value)