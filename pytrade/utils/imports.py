import importlib
import importlib.util
from inspect import getmembers, isfunction
from pathlib import Path
from typing import List, Callable


def py_obj_from_str(name):
    module_name, obj_name = name.rsplit(".", 1)
    module = importlib.import_module(module_name)
    return getattr(module, obj_name)


def get_module_from_path(module_path):
    module_name = Path(module_path).stem
    spec = importlib.util.spec_from_file_location(module_name, module_path)
    module = importlib.util.module_from_spec(spec)
    # sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def get_obj_from_path(obj_path):
    module_path, obj_name = obj_path.split(":")
    module = get_module_from_path(module_path)
    return getattr(module, obj_name)


def get_all_modules(path: str):
    """
    Notes
    -----
    https://docs.python.org/3/library/importlib.html#importing-a-source-file-directly
    """
    modules = []
    for path in Path(path).rglob('*.py'):
        if path.name != "__init__.py":
            spec = importlib.util.spec_from_file_location(path.stem, path)
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            modules.append(module)
    return modules


def get_module_functions(m):
    """
    Only gets functions defined in the module.
    """
    return [x[1] for x in getmembers(m, isfunction) if
            x[1].__module__ == m.__name__]


def get_functions_with_attr(path: str, attr: str) -> List[Callable]:
    modules = get_all_modules(path)
    return [x for m in modules for x in get_module_functions(m) if hasattr(x, attr)]
