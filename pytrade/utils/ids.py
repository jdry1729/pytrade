import hashlib


def get_full_obj_name(name: str, version: str):
    return f"{name}_v{version}"


def get_obj_id(name: str, version: str):
    sha = hashlib.sha256()
    qualified_model_name = get_full_obj_name(name, version)
    sha.update(qualified_model_name.encode())
    return sha.hexdigest()
