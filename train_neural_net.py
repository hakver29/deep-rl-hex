from os import listdir
from os.path import isfile, join

from GameSetting import GameSetting
from Policy import Policy
from definitions import DATA_DIR


def import_data(file_name):
    lines = open(DATA_DIR+file_name)
    features = []
    targets = []
    for line in lines:
        sectors = line.split("|")
        features.append([ float(x) for x in sectors[0].split(",")])
        targets.append([float(x) for x in sectors[1].split(",")])

    return features, targets

def read_training_data(size):
    path = DATA_DIR
    files_with_training_data = [f for f in listdir(path) if isfile(join(path, f))]
    features = targets = []
    for file_name in files_with_training_data:
        layer_dims = [int(x) for x in file_name.split("-")[0].split('n')]
        if layer_dims[0] == game_setting.size ** 2 and layer_dims[len(layer_dims) - 1] == game_setting.size ** 2:
            training_data = import_data(file_name)
            features = features + training_data[0]
            targets = targets + training_data[1]

    return features, targets

game_setting = GameSetting()
policy = Policy(game_setting)
x,y = read_training_data(game_setting.size)
policy.train(x, y)