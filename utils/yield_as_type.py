def yield_as(t_yield: type, item_if_single: bool=False):
    def decorator(func):
        def wrapper(*args, **kwargs):
            items = t_yield(func(*args, **kwargs))
            if item_if_single:
                match len(items):
                    case 0: return None
                    case 1: return items[0]
            return items
        return wrapper
    return decorator
