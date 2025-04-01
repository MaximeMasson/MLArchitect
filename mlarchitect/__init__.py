import json
import os
from pathlib import Path

mlarchitect_config = {}

def load_config(profile, filepath=""):
    filename = os.path.join(filepath, 'projects/projectsConfig.json')
    global mlarchitect_config
    if not filename:
        home = str(Path.home())
        filename = os.path.join(home, ".projectsConfig.json")
        if not os.path.isfile(filename):
            raise Exception(f"If no 'filename' parameter specified, assume '.projectsConfig.json' exists at HOME: {home}")

    with open(filename) as f:
        data = json.load(f)
        if profile not in data:
            raise Exception(f"Undefined profile '{profile}' in file '{filename}'")
        mlarchitect_config = data[profile]