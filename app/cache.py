# Cache en memoria compartida entre módulos
# Se limpia cuando Render redespliega, pero dura toda la sesión activa

_cache = {}


def set(key: str, value) -> None:
    """Almacena un valor en el cache."""
    _cache[key] = value


def get(key: str):
    """Obtiene un valor del cache. Retorna None si no existe."""
    return _cache.get(key)


def delete(key: str) -> None:
    """Elimina una clave del cache."""
    _cache.pop(key, None)


def keys():
    """Retorna todas las claves del cache."""
    return list(_cache.keys())


def clear() -> None:
    """Limpia todo el cache."""
    _cache.clear()


def exists(key: str) -> bool:
    """Verifica si una clave existe en el cache."""
    return key in _cache