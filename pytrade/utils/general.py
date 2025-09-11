from dataclasses import dataclass


@dataclass
class Progress:
    done: int
    total: int


def get_attrs(obj):
    """
    Returns a dict of all attributes of the object, and all dynamic properties.
    """
    attrs = {}
    for k, v in vars(obj).items():
        if not k.startswith("_"):
            attrs[k] = v
    for k, v in vars(obj.__class__).items():
        if isinstance(v, property):
            attrs[k] = getattr(obj, k)
    return attrs
