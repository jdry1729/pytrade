import math
import secrets


def generate_uid(n: int = 12) -> str:
    return secrets.token_bytes(math.ceil(n / 2.0)).hex()[:n]
