def short_repr(obj, max_len=3, max_depth=2):
    if isinstance(obj, dict):
        N = len(obj)
        out = "{"
        for i, (k, v) in enumerate(obj.items()):
            out += f"'{k}': {short_repr(v, max_len=max_len)}"
            if i < N - 1:
                out += ", "
        out += "}"
    elif isinstance(obj, list):
        out = "["
        N = len(obj)
        for i, x in enumerate(obj):
            if i >= max_len:
                out += "..."
                break
            out += short_repr(x)
            if i < N - 1:
                out += ", "
        out += "]"
    else:
        out = repr(obj)
    return out
