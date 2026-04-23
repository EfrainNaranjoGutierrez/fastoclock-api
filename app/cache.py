# Cache en memoria compartida entre módulos
# Se limpia cuando Render redespliega, pero dura toda la sesión activa

_cache = {}


def set(key: str, value) -> None:
    _cache[key] = value


def get(key: str):
    return _cache.get(key)


def delete(key: str) -> None:
    _cache.pop(key, None)


def keys():
    return list(_cache.keys())
