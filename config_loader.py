import json

def load_config(filepath='config.json'):
    with open(filepath, 'r') as config_file:
        return json.load(config_file)
