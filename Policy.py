import random
import numpy as np
import tensorflow as tf
from os import listdir
from os.path import isfile, join
import sys

from GameSetting import GameSetting
from definitions import DATA_DIR

class Policy:

    def __init__(self, game_setting):
        self.game_setting = game_setting
        dims = [str(int) for int in game_setting.network_dimensions]
        afunc = eval("tf.nn." + game_setting.activation_function)
        hfunc = eval("tf.nn." + game_setting.hidden_function)
        ofunc = eval("tf.nn." + game_setting.output_function)

        self.model = tf.keras.models.Sequential() #Defining feed-forward neural net
        self.model.add(tf.keras.layers.Dense(dims[0], input_shape=(dims[0],), activation=afunc)) #Add input layer
        for i in range(1,len(dims)-1):
            self.model.add(tf.keras.layers.Dense(dims[i], activation=hfunc)) #add hidden layers
        self.model.add(tf.keras.layers.Dense(dims[len(dims)-1], activation=ofunc)) #add output layer

        #self.model.add(tf.keras.layers.Dense(

        optimizer = eval("tf.keras.optimizers."+game_setting.optimizer+"(lr="+str(game_setting.learning_rate)+")")

        self.model.compile(optimizer=optimizer, loss=game_setting.loss_function, metrics=[game_setting.metrics])

    def select(self, feature_vector, legal_moves, stochastic=False):
        feature_vector = np.array(feature_vector)
        loopend = feature_vector.shape[0]
        feature_vector = np.expand_dims(feature_vector, 0)
        probability_of_moves = self.model.predict_on_batch(feature_vector)
        for i in range(0,loopend):
            if i not in legal_moves:
                probability_of_moves[0,i] = -10**6 #Removing all non-legal moves from neural net prediction
        if stochastic:
            pass
        else:
            return probability_of_moves.argmax() #Returning the move with highest probability score

        #probability_of_moves = probability_of_moves/probability_of_moves.sum() #Adjusting all remaining probabilities

    def read_all_training_data(self):
        path = DATA_DIR
        files_with_training_data = [f for f in listdir(path) if isfile(join(path, f))]
        features = []
        targets = []
        for file_name in files_with_training_data:
            layer_dims = [int(x) for x in file_name.split("-")[0].split('n')]
            if layer_dims[0] == self.game_setting.size ** 2 and layer_dims[len(layer_dims) - 1] == self.game_setting.size ** 2:
                training_data = self.import_data_from_single_file(file_name)
                features = features + training_data[0]
                targets = targets + training_data[1]

        assert len(features) > 10
        assert len(targets) > 10

        return features, targets

    def import_data_from_single_file(self, file_name):
        lines = open(DATA_DIR + file_name)
        features = []
        targets = []
        for line in lines:
            sectors = line.split("|")
            features.append([float(x) for x in sectors[0].split(",")])
            targets.append([float(x) for x in sectors[1].split(",")])

        return features, targets

    def train(self, feature_vectors, targets, max_cases, batch_size=10):
        assert (len(feature_vectors) == len(targets))
        feature_vectors = np.array([np.array(feature_vector) for feature_vector in feature_vectors])
        targets = np.array([np.array(target) for target in targets])

        nr_of_cases = min(max_cases, int((self.game_setting.case_fraction*len(feature_vectors))//2))
        indexes = np.random.choice(feature_vectors.shape[0], nr_of_cases, replace=False)

        feature_vectors = feature_vectors[indexes, :]
        targets = targets[indexes, :]

        self.model.fit(feature_vectors, targets, epochs=self.game_setting.epochs, batch_size=batch_size)
        #tf.keras.utils.plot_model(self.model, to_file='model.png')
        return nr_of_cases

    def import_data_and_train(self, max_cases=sys.maxsize):
        features, targets = self.read_all_training_data()
        return self.train(features, targets, max_cases)