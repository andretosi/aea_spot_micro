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

            #Test Stub1 state
            self.assertEqual(s1.p1, 0.5)
            self.assertEqual(s1.p2, "param")
            self.assertEqual(s1.p3, 2)

            #Test config state
            #self.assertEqual(cfg.central_registry[s1.__class__.__name__], cfg_dict) <- fix?

        finally:
            os.remove(path)