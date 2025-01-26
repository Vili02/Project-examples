import os
import importlib

def load_modules():
    modules = {}
    module_dir = os.path.dirname(__file__)
    for file in os.listdir(module_dir):
        if file.endswith(".py") and file != "__init__.py":
            module_name = file[:-3]
            module = importlib.import_module(f"modules.{module_name}")
            modules[module_name] = module
    return modules
