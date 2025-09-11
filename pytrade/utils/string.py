import re
import string
from typing import Dict


def replace(s, to_replace: Dict):
    for k, v in to_replace.items():
        s = s.replace(k, v)
    return s


def snake_to_title(s: str) -> str:
    return " ".join(s.split("_")).title()


def snake_to_pascal(s: str) -> str:
    return "".join(x.capitalize() for x in s.split("_"))


def title_to_snake(s: str) -> str:
    s = re.sub(r"[^a-zA-Z0-9\s]", "", s)
    return "_".join(s.lower().split())


def str_to_num(s: str):
    try:
        return int(s)
    except Exception:
        try:
            return float(s)
        except Exception:
            return s


def get_format_args(x, named_only=True):
    args = set([tup[1] for tup in string.Formatter().parse(x) if tup[1] is not None])
    if named_only:
        args = [x for x in args if x != ""]
    return set(args)
