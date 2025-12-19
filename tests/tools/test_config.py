from spotmicro.tools.config import Config

import unittest, tempfile, yaml, os

"""
This file should test Config as a standalone class, without considering its interactions with the "configurable" decorator. Therefore, parameters for the Dummy class are to be treated as dictionaries, since Config works with them
"""

class Dummy:
    pass

class TestConfigBasic(unittest.TestCase):
    def test_empty_construction(self):
        config = Config()
        self.assertEqual(config.central_registry, {})
        self.assertEqual(config.registered_objects, [])
    
    def test_construction_from_file(self):
        data = {
            "Dummy": {
                "a": 1,
                "b": 3
            }
        }

        with tempfile.NamedTemporaryFile(mode="w", delete=False) as f:
            yaml.safe_dump(data, f)
            path = f.name

        try:
            cfg = Config(path)
            self.assertEqual(cfg.central_registry, data)
        finally:
            os.remove(path)
    
    def test_register_new_component(self):
        cfg = Config()
        obj = Dummy()

        params = {"a": 1, "b": 2}
        returned = cfg.register(Dummy, obj, params)

        self.assertEqual(returned, {})
        self.assertIn(obj, cfg.registered_objects)
        self.assertEqual(len(cfg.registered_objects), 1)
        self.assertEqual(cfg.central_registry["Dummy"], params)
    

    def test_register_overrides_file_params(self):
        initial = {
            "Dummy": {
                "a": 1,
                "b": 2
            }
        }

        with tempfile.NamedTemporaryFile(mode="w", delete=False) as f:
            yaml.safe_dump(initial, f)
            path = f.name

        try:
            cfg = Config(path)
            obj = Dummy()

            overrides = {"b": 99}
            remaining = cfg.register(Dummy, obj, overrides)

            self.assertEqual(remaining, {"a": 1})
            self.assertEqual(cfg.central_registry["Dummy"], {"a": 1, "b": 99})
        finally:
            os.remove(path)
    

    def test_update(self):
        cfg = Config()
        obj = Dummy()

        cfg.register(Dummy, obj, {"a": 1})
        cfg.update(obj, {"a": 2, "c": 3})

        self.assertEqual(
            cfg.central_registry["Dummy"],
            {"a": 2, "c": 3}
        )

    def test_save(self):
        cfg = Config()
        obj = Dummy()
        cfg.register(Dummy, obj, {"x": 10})

        with tempfile.NamedTemporaryFile(mode="r", delete=False) as f:
            path = f.name

        try:
            cfg.save(path)

            with open(path, "r") as f:
                data = yaml.safe_load(f)

            self.assertEqual(data, {"Dummy": {"x": 10}})
        finally:
            os.remove(path)