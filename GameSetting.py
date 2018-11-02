import yaml
from definitions import ROOT_DIR

class GameSetting:
    def __init__(self):
        configfile = yaml.load(open(ROOT_DIR+"WhichSetting.yaml"))
        config = yaml.load(open(ROOT_DIR+configfile["filename"]))
        self.G = config["G"]
        self.P = config["P"]
        self.M = config["M"]
        self.verbose = config["verbose"]
        self.size = config["size"]