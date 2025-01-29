import pprint
class Config:
    def __init__(self, config_dict):

        for key, value in config_dict.items():
            if isinstance(value, dict):  # Recursively handle nested dictionaries
                setattr(self, key, Config(value))
            else:
                setattr(self, key, value)
        
    def __repr__(self):
        return f"Config({pprint.pformat(self.__dict__, indent=4)})"

    def __iter__(self):
        # Yield key-value pairs for the class attributes
        for key, value in self.__dict__.items():
            yield key, dict(value) if type(value) is Config else value
