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


class Config(metaclass=SingletonMeta):
    def __init__(self):
        # Initialize configuration settings
        self.settings = {}
        self.load_settings()

    def load_settings(self):
        # Load settings from a file or environment variables
        self.settings = {
            'database_url': 'localhost:5432',
            'api_key': '1234567890abcdef',
            'debug': True
        }

    def get_setting(self, key):
        return self.settings.get(key)


# Usage
config1 = Config()
config2 = Config()

print(config1 is config2)  # Output: True
print(config1.get_setting('database_url'))  # Output: localhost:5432
