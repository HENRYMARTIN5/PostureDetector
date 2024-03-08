import json


class NotifierConfig:
    def __init__(self, config={"enabled": False, "id": "changeme", "name": "CHANGEME!", "description": "CHANGEME!"}, source_config=None):
        """
        Initializes a new NotifierConfig object
        :param config: The configuration to use
        :param source_config: A tuple containing the source config and the key to use in the source config
        """
        self.config = config
        self.source_config = source_config
        if source_config:
            keys = source_config[0].keys()
            if source_config[1] not in keys:
                source_config[0].set(source_config[1], config)
            else:
                self.config = source_config[0].get(source_config[1])

    def get(self, key, default=None):
        """
        Gets a value from the config
        :param key: The key to get
        :param default: The default value to return if the key is not found
        :return: The value of the key, or the default value if the key is not found
        """
        return self.config.get(key, default)

    def set(self, key, value):
        """
        Sets a value in the config
        :param key: The key to set
        :param value: The value to set
        """
        self.config[key] = value
        if self.source_config:
            old = self.source_config[0].get(self.source_config[1])
            old[key] = value
            self.source_config[0].set(self.source_config[1], old)

    def create(self, key, default_value, pretty_name, description, type="text"):
        """
        Creates a new user-editable config value
        :param key: The internal key name
        :param default_value: The default value
        :param pretty_name: The pretty name to display in the webui
        :param description: The description to display in the webui
        :param type: The type of the value (HTML form type, e.g. "text", "number", "checkbox", etc.)
        :return: True if the value was created, False if it already exists
        """
        if key in self.config:
            return False
        new_entry = {
            "default": default_value,
            "name": pretty_name,
            "description": description,
            "value": default_value,
            "type": type
        }
        self.config[key] = new_entry
        if self.source_config:
            old = self.source_config[0].get(self.source_config[1])
            old[key] = new_entry
            self.source_config[0].set(self.source_config[1], old)

    def get_user(self, key):
        """
        Gets a user-editable config value
        :param key: The key to get
        :return: The value of the key
        """
        return self.config[key]["value"]


class Notifier:
    def __init__(self, config: NotifierConfig):
        """
        Initializes a new Notifier object
        :param config: The NotifierConfig object to use. Provided by the host application
        """
        self.config = config

    def notify(self, posture):
        """
        This method is called when a posture alert is triggered
        :param posture: The posture (as an int) that was detected
        """
        pass

    def _notify(self, posture):
        """
        Internally used. DO NOT OVERRIDE!
        """
        if self.config.get('enabled'):
            self.notify(posture)
    
    def reload_config(self):
        """
        This method is called when the config is modified.
        """
        self.config = NotifierConfig(source_config=(self.config.source_config[0], "notifiers." + self.config.get("id")))

    def __str__(self):
        return json.dumps(self.config.getall())

    def __repr__(self):
        return self.__str__()
