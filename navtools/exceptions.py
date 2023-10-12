from dataclasses import fields


class InvalidReceiverSimulationConfiguration(Exception):
    def __init__(self, config):
        config_fields = fields(config)
        invalid_fields = [
            field.name for field in config_fields if type(field.default) is None
        ]

        message = f"populate the following parameters: {invalid_fields}"
        self.message = message
        super().__init__(self.message)
