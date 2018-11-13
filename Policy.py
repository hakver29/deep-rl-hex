import tensorflow as tf
import numpy as np
import random

class Policy:

    def __init__(self, gamesetting):
        self.gamesetting = gamesetting
        dims = gamesetting.network_dimensions
        afunc = gamesetting.activation_function
        hfunc = gamesetting.hidden_function
        ofunc = gamesetting.output_function

        self.model = tf.keras.models.Sequential() #Defining feed-forward neural net
        self.model.add(tf.keras.layers.Flatten()) #Add "flat" input layer
        exec("self.model.add(tf.keras.layers.Dense("+dims[0]+", activation=tf.nn."+afunc+"))") #Add input layer
        for i in range(1,len(dims)-1):
            exec("self.model.add(tf.keras.layers.Dense(" + dims[i] + ", activation=tf.nn." + hfunc + "))") #add hidden layers
        exec("self.model.add(tf.keras.layers.Dense(" + dims[len(dims)] + ", activation=tf.nn." + ofunc + "))") #add output layer

        self.model.compile(optimizer=gamesetting.optimizer,loss=gamesetting.loss_function, metrics=gamesetting.metrics)

    def select(self, feature_vector, legal_moves):
        probability_of_moves = self.model.predict(feature_vector)
        probability_of_moves = np.array(probability_of_moves)
        for i in range(0,len(probability_of_moves)):
            if i not in legal_moves:
                probability_of_moves[i] = 0 #Removing all non-legal moves from neural net prediction

        return random.choice(probability_of_moves.argmax()) #Returning the move with highest probability score
                                                            #If several moves have equal probability, return random

        #probability_of_moves = probability_of_moves/probability_of_moves.sum() #Adjusting all remaining probabilities



    def train(self, feature_vectors, targets):
        assert (len(feature_vectors) == len(targets))
        self.model.fit(feature_vectors,targets,epochs=self.gamesetting['epochs'])