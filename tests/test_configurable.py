from spotmicro.tools.config import Config
from spotmicro.tools.configurable import configurable

import unittest, tempfile, yaml, os

@configurable
class Stub1:
    """
    SUCCESSFULL Stub1
    """

    def __init__(self, config: Config, p1=0.0, p2="param", p3=1):
        self.p1 =  p1
        self.p2 = p2
        self.p3 = p3

class stub1TestCase(unittest.TestCase):

    #<----- BASIC FUNCTIONALITIES ----->
    def test_normal_construction(self):
        emptyConfig = Config()
        s1 = Stub1(emptyConfig)

        self.assertEqual(s1.p1, 0.0, "param mismatch")
        self.assertEqual(s1.p2, "param", "param mismatch")
        self.assertEqual(s1.p3, 1, "param mismatch")
        self.assertIn(s1.__class__.__name__, emptyConfig.central_registry, "Stub was not registered correctly")
    
    def test_build_from_config(self):
        cfg_dict = {
            "Stub1": {
                "p1": 0.5,
                "p3": 2
            }
        }

        with tempfile.NamedTemporaryFile(mode="w", delete=False) as f:
            yaml.safe_dump(cfg_dict, f)
            path = f.name
        
        try:
            cfg = Config(path)
            s1 = Stub1(cfg)

            #Test Stub1 state: are parameters set correctly?
            self.assertEqual(s1.p1, 0.5)
            self.assertEqual(s1.p2, "param")
            self.assertEqual(s1.p3, 2)

            #Test config state
            self.assertEqual(cfg.central_registry, cfg_dict) #Does central_registry reflect the real config_dictionary?
            self.assertEqual(cfg.registered_objects, [s1]) #Was the object registered correctly?

        finally:
            os.remove(path)
    
    def test_load(self):
        cfg = Config()
        s1 = Stub1(cfg, p1=1.5, p3=4)

        cfg_dict = {
            "Stub1": {
                "p1": 2.0,
                "p2": "pepepe"
            }
        }

        expected_registry_dict = {
            "Stub1": {
                "p1": 2.0,
                "p2": "pepepe",
                "p3": 4
            }
        }

        with tempfile.NamedTemporaryFile(mode="w", delete=False) as f:
            yaml.safe_dump(cfg_dict, f)
            path = f.name
        try:
            s1.load(path)

            #Were the parameters set correctly? Expected behaviour is that load overrides constructor parameters
            self.assertEqual(s1.p1, 2.0)
            self.assertEqual(s1.p2, "pepepe")
            self.assertEqual(s1.p3, 4)

            #Is config state valid?
            self.assertEqual(cfg.central_registry, expected_registry_dict)
            self.assertEqual(cfg.registered_objects, [s1])
        finally:
            os.remove(path)
    
    def test_save(self):
        # Create a config with some preloaded parameters
        cfg_dict = {
            "Stub1": {
                "p2": "config_value"
            }
        }

        with tempfile.NamedTemporaryFile(mode="w", delete=False) as f:
            yaml.safe_dump(cfg_dict, f)
            path = f.name

        try:
            cfg = Config(path)
            # Instantiate with a constructor override and leave one param as default
            s1 = Stub1(cfg, p1=1.5)

            # Save the component's config
            save_path = path + "_out"
            s1.save(save_path)

            # Load back the file to check correctness
            with open(save_path, "r") as f:
                saved_data = yaml.safe_load(f)

            # Expected: constructor param (p1) overrides default, config param (p2) preserved, default param (p3) not in config
            expected_data = {
                "Stub1": {
                    "p1": 1.5,
                    "p2": "config_value"
                }
            }

            self.assertEqual(saved_data, expected_data)
            self.assertEqual(cfg.registered_objects, [s1])
        finally:
            os.remove(path)
            if os.path.exists(save_path):
                os.remove(save_path)

    #<----- EDGE CASES ----->