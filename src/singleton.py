class SingletonMeta(type):
    """
    This is a thread-safe implementation of Singleton.
    """
    _instances = {}

    def __call__(cls, *args, **kwargs):
        if cls not in cls._instances:
            print(f"Creating new instance for {cls.__name__}")
            cls._instances[cls] = super(SingletonMeta, cls).__call__(*args, **kwargs)
        else:
            print(f"Using existing instance for {cls.__name__}")
        return cls._instances[cls]
