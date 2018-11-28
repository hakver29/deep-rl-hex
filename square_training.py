from definitions import DATA_DIR
import numpy as np

file_name = DATA_DIR+"50n49-current_max"
power = 5
scale_features = True
output_file_name = file_name+"_transformed_with_power_"+str(power)
if scale_features:
    output_file_name += "-features_scaled"
output_file = open(output_file_name, "w+")


def import_data_from_single_file(file_name):
    lines = open(file_name)
    features = []
    targets = []
    for line in lines:
        sectors = line.split("|")
        features.append([float(x) for x in sectors[0].split(",")])
        targets.append([float(x) for x in sectors[1].split(",")])

    return features, targets


def raise_targets_to_power(targets, power):
    for i in range(0, len(targets)):
        #print("Original target: " + str(targets[i]))
        for k in range(0, len(targets[i])):
            targets[i][k] = targets[i][k]**power
        targets[i] = np.array(targets[i])
        targets[i] = targets[i]/targets[i].sum()
        #print("Transformed target: " + str(targets[i]))
    return targets

def reassign_features(features):
    for i in range(0, len(features)):
        for k in range(0, len(features[i])):
            if features[i][k] == 1:
                features[i][k] = -100
            elif features[i][k] == 2:
                features[i][k] = 100
    return features

features, targets = import_data_from_single_file(file_name)

targets = raise_targets_to_power(targets, 5)

if scale_features:
    features = reassign_features(features)

assert len(features) == len(targets)
for i in range(0, len(features)):
    feature = features[i]
    target = targets[i]
    output_file.write(",".join(str(int(input)) for input in feature)+"|"+",".join(str(target) for target in target)+"\n")