import random
import numpy as np
import tensorflow as tf
from os import listdir
from os.path import isfile, join
import time
import sys
import warnings

from GameSetting import GameSetting
from definitions import DATA_DIR, MODEL_DIR, ROOT_DIR, REINFORCEMENT_MODEL_DIR



class Policy:

    def __init__(self, game_setting):
        self.game_setting = game_setting
        dims = [str(int) for int in game_setting.network_dimensions]
        afunc = eval("tf.nn." + game_setting.activation_function)
        hfunc = eval("tf.nn." + game_setting.hidden_function)
        ofunc = eval("tf.nn." + game_setting.output_function)

        self.model = tf.keras.models.Sequential() #Defining feed-forward neural net
        self.model.add(tf.keras.layers.Dense(dims[0], input_shape=(dims[0],), activation=afunc, kernel_initializer=
tf.keras.initializers.Ones())) #Add input layer
        for i in range(1,len(dims)-1):
            self.model.add(tf.keras.layers.Dense(dims[i], activation=hfunc)) #add hidden layers
        self.model.add(tf.keras.layers.Dense(dims[len(dims)-1], activation=ofunc)) #add output layer

        #self.model.add(tf.keras.layers.Dense(

        optimizer = eval("tf.keras.optimizers."+game_setting.optimizer+"(lr="+str(game_setting.learning_rate)+")")

        self.model.compile(optimizer=optimizer, loss=game_setting.loss_function, metrics=[game_setting.metrics])

    def scale_probabilities(self, vector, power):
        for i in range(0, len(vector)):
            vector[i] = vector[i] ** power

        return vector

    def stochastic_selection(self, probability_of_moves, legal_moves):
        random_num = random.uniform(0, 1)
        sum = 0.0

        for i in range(0, probability_of_moves.shape[1]):
            if sum + probability_of_moves.item(i) > random_num:
                if self.game_setting.verbose >= 3:
                    print("Selected move had " + str(probability_of_moves.item(i) * 100)[
                                                 0:4] + "% probability of being selected.")
                return i
            if i > 0 and sum + probability_of_moves.item(i) >= random_num:
                if self.game_setting.verbose >= 3:
                    print("Selected move had " + str(probability_of_moves.item(i) * 100)[
                                             0:4] + "% probability of being selected.")
                return i
            sum += probability_of_moves.item(i)

        warnings.warn("All moves had zero probability in neural net prediction.")

        if self.game_setting.verbose >= 3:
            print("RANDOMLY selected move. All moves had zero probability in prediction.")
        return random.choice(legal_moves)

    def select(self, feature_vector, legal_moves):
        feature_vector = np.array(feature_vector)
        loopend = feature_vector.shape[0]-1
        feature_vector = np.expand_dims(feature_vector, 0)
        probability_of_moves = self.model.predict(feature_vector)
        printable_vector = [(i) for i in probability_of_moves[0]]
        if self.game_setting.verbose >= 3:
            print(printable_vector)
        for i in range(0,loopend):
            if i not in legal_moves:
                probability_of_moves[0,i] = 0.0 #Removing all non-legal moves from neural net prediction
        if self.game_setting.stochastic:
            if self.game_setting.raise_probabilities_power > 1:
                probability_of_moves = self.scale_probabilities(probability_of_moves, self.game_setting.raise_probabilities_power) #Squaring all probabilities before
                                                                                #stochastic selection
            probability_of_moves = probability_of_moves/probability_of_moves.sum()
            return self.stochastic_selection(probability_of_moves, legal_moves)
        else:
            for i in range(0, loopend):
                if i not in legal_moves:
                    probability_of_moves[0, i] = -1000.0  # Removing all non-legal moves from neural net prediction
            return probability_of_moves.argmax() #Returning the move with highest probability score

            #probability_of_moves = probability_of_moves/probability_of_moves.sum() #Adjusting all remaining probabilities

    def read_all_training_data(self, training_file=None):
        path = DATA_DIR
        files_with_training_data = [f for f in listdir(path) if isfile(join(path, f))]
        features = []
        targets = []
        if training_file is not None:
            training_data = self.import_data_from_single_file(training_file)
            features = features + training_data[0]
            targets = targets + training_data[1]
            return features, targets
        else:
            for file_name in files_with_training_data:
                layer_dims = [int(x) for x in file_name.split("-")[0].split('n')]
                if layer_dims[0] == self.game_setting.nr_of_legal_moves+1 and layer_dims[len(layer_dims) - 1] == self.game_setting.nr_of_legal_moves:
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

    def train(self, feature_vectors, targets, max_cases, batch_size=10, test_nr_of_cases=False):
        assert (len(feature_vectors) == len(targets))
        feature_vectors = np.array([np.array(feature_vector) for feature_vector in feature_vectors])
        targets = np.array([np.array(target) for target in targets])

        nr_of_cases = min(max_cases, int((self.game_setting.case_fraction*len(feature_vectors))))
        if test_nr_of_cases:
            return nr_of_cases

        indexes = np.random.choice(feature_vectors.shape[0], nr_of_cases, replace=False)

        feature_vectors = feature_vectors[indexes, :]
        targets = targets[indexes, :]

        self.model.fit(feature_vectors, targets, epochs=self.game_setting.epochs, batch_size=batch_size, shuffle=False)
        if self.game_setting.display_summary:
            self.model.summary()

        #tf.keras.utils.plot_model(self.model, to_file='model.png')
        #self.model.save("model")
        return nr_of_cases

    def import_data_and_train(self, max_cases=sys.maxsize, training_file=None, test_nr_of_cases=False):
        features, targets = self.read_all_training_data(training_file=None)
        return self.train(features, targets, max_cases, test_nr_of_cases=test_nr_of_cases)

    def load_best_model(self):

        file_path_best_model, nr__of_training_cases = self.find_file_path_of_best_model()
        self.model = tf.keras.models.load_model(file_path_best_model)
        if self.game_setting.verbose >= 1:
            print("Loaded best model from storage.")
        return nr__of_training_cases

    def find_file_path_of_best_model(self):
        files_with_correct_dimensions = []

        path = MODEL_DIR
        files_with_training_data = [f for f in listdir(path) if isfile(join(path, f))]
        for file_name in files_with_training_data:
            input_size = int(file_name.split("-")[0].split('n')[0])
            if input_size == self.game_setting.nr_of_legal_moves+1:
                files_with_correct_dimensions.append(file_name)

        highest_win_rate = 0.0
        highest_win_rate_file = 0
        training_cases = 0

        for file_name in files_with_correct_dimensions:
            win_rate = float(file_name.split("-")[1])
            if win_rate>=highest_win_rate:
                highest_win_rate = win_rate
                highest_win_rate_file = file_name
                training_cases = int(file_name.split("-")[2])

        if highest_win_rate_file == 0:
            raise ValueError("Could not find file with highest win rate.")

        return MODEL_DIR+highest_win_rate_file, training_cases

    def load_reinforcement_model(self, i):
        reinforcement_model_files = []

        path = REINFORCEMENT_MODEL_DIR

        files_with_training_data = [f for f in listdir(path) if isfile(join(path, f))]

        for file_name in files_with_training_data:
            reinforcement_model_files.append(file_name)

        reinforcement_model_files = sorted(reinforcement_model_files, key=lambda k: int(k.split("-")[2]))

        if i > len(reinforcement_model_files)-1:
            raise ValueError("Tried to load reinforcement models without saved models in " + REINFORCEMENT_MODEL_DIR)

        self.model = tf.keras.models.load_model(path+reinforcement_model_files[i])
        return reinforcement_model_files[i]
