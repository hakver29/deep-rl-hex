#from Node import *
#from HexState import *
from Node import *
from HexState import *
from GameSetting import *
import copy
from Policy import *
import train_neural_net
from definitions import MODEL_DIR
import os
from hexclient.BasicClientActor import BasicClientActor as BSA
import time

def convertIntegerToCoordinate(intMove, boardSize):
    ycoordinate = intMove//boardSize
    xcoordinate = intMove%boardSize
    return xcoordinate,ycoordinate


def convertCoordinateToInteger(move, boardSize):
    return move[1]*boardSize + move[0]

def tree_search(rootstate, itermax, verbose=False, policy=None, policies=None, save_training=True):
    rootnode = Node1(state=rootstate)
    for i in range(itermax):
        node = rootnode
        state = copy.deepcopy(rootstate)

        """
        Selection: start from root R and select successive child nodes until a leaf node L is reached. The root is the 
        current game state and a leaf is any node from which no simulation (playout) has yet been initiated. The 
        section below says more about a way of biasing choice of child nodes that lets the game tree expand towards the 
        most promising moves, which is the essence of Monte Carlo tree search.

        Expansion: unless L ends the game decisively (e.g. win/loss/draw) for either player, create one (or more) child 
        nodes and choose node C from one of them. Child nodes are any valid moves from the game position defined by L.

        Simulation: complete one random playout from node C. This step is sometimes also called playout or rollout. A 
        playout may be as simple as choosing uniform random moves until the game is decided (for example in chess, the 
        game is won, lost, or drawn).

        Backpropagation: use the result of the playout to update information in the nodes on the path from C to R.

        -Wikipedia
        """

        # Selection
        while node.untried_moves == [] and node.childNodes != []:
            node = node.select_child()
            state.play(node.move)

        # Expansion
        if node.untried_moves != []:
            move = random.choice(node.untried_moves)
            state.play(move)
            node = node.add_child(move, state)

        # Simulation
        if policies is None:
            # Simulation with random, vanilla MCTS if no neural net policy is defined.
            while state.winner() == 0:
                state.play(random.choice(state.moves()))
        else:
            #If a neural net policy is defined, we let the neural net do the rollouts / simulations
            while state.winner() == 0:
                random_num = random.uniform(0, 1)
                #If our random number exceeds epsilon, we let the ANN pick move. If not, the move is random.
                if random_num>game_setting.epsilon:
                    legal_moves = [state.convertCoordinateToInteger(move) for move in state.moves()]
                    flattened_move = policies[state.toplay-1].select(state.convertFeatureVectorToFormat(rootstate.board.flatten('F'), rootstate.toplay),
                                                     legal_moves)
                    assert (flattened_move in legal_moves)
                    state.play(state.convertIntegerToCoordinate(flattened_move))
                else:
                    state.play(random.choice(state.moves()))

        # Backpropagation
        while node != None:
            node.visits += 1
            if node.toplay != state.winner():
                node.wins += 1
            node = node.parentNode

    if game_setting.verbose == True:
        print(rootnode.children_to_string())

    if save_training:
        append_result_to_training_data(rootnode, rootstate)
    return max(rootnode.childNodes, key=lambda c: c.visits).move


def play_game(game_setting, policies=None, bad_mcts=False, bad_vs_good_neural_net=None):
    """
    Spiller et enkelt spill mellom to spillere
    """
    #Bad mcts mode pits a strong mcts player versus a weak mcts player. The idea is make
    #training data that teaches ANNs how to not only play vs strong players, but weak
    #players too. Weak players make random moves in odd corners of the board that the
    #ANN may respond erratically to.
    player_wins = {"black": 0, "white": 0}
    for i in range(game_setting.G):
        state = HexState1(game_setting)
        while (state.white_groups.connected(1,2) == False and state.black_groups.connected(1,2) == False):
            if bad_mcts:
                if state.toplay == 2:
                    move = tree_search(rootstate=state, itermax=game_setting.M, verbose=game_setting.verbose,
                                       policies=None, save_training=True)
                elif state.toplay == 1:
                    move = tree_search(rootstate=state, itermax=game_setting.M//15, verbose=game_setting.verbose,
                                       policies=None, save_training=False)
            elif bad_vs_good_neural_net is not None:
                #Bad neural net on first index, good neural net on second index
                if state.toplay == 2:
                    move = tree_search(rootstate=state, itermax=game_setting.M//2, verbose=game_setting.verbose,
                                       policies=policies, save_training=True)
                elif state.toplay == 1:
                    move = tree_search(rootstate=state, itermax=game_setting.M//30, verbose=game_setting.verbose,
                                       policies=policies, save_training=False)
            else:
                move = tree_search(rootstate=state, itermax=game_setting.M, verbose=game_setting.verbose, policies=policies)
            if state.toplay == 2:
                state.place_black(move)
                state.set_turn(1)
            elif state.toplay == 1:
                state.place_white(move)
                state.set_turn(2)

            if game_setting.verbose == True:
                print(state)
                if state.toplay == 1:
                    print("Player 2 selects " + str(move) + "\n")
                elif state.toplay == 2:
                    print("Player 1 selects " + str(move) + "\n")
                #print("Player " + str(state.toplay) + " selects " + str(move))
        if game_setting.verbose == True:
            if state.winner() == 2:
                print("Player black wins" + "\n")
            elif state.winner() == 1:
                print("Player white wins" + "\n")

        print(state)

        if state.winner() == 1:
            player_wins["white"] += 1
        elif state.winner() == 2:
            player_wins["black"] += 1
    print(player_wins)



def append_result_to_training_data(rootnode, rootstate):
    target = [0] * game_setting.nr_of_legal_moves

    for child_node in rootnode.childNodes:
        move = child_node.move[1]*game_setting.size+child_node.move[0]
        target[move] = child_node.visits/rootnode.visits

    feature_vector = rootstate.convertFeatureVectorToFormat(rootstate.board.flatten('F'),rootstate.toplay)

    training_data_file.write(",".join(str(int(input)) for input in feature_vector)+"|"+",".join(str(target) for target in target)+"|"+"\n")

start_time = time.time() #We start counting the time.

game_setting = GameSetting() #Load the game settings.

file_path = training_data_file_path = DATA_DIR+'n'.join(str(dim) for dim in game_setting.network_dimensions)+"-"+str(time.time()+datetime.now().microsecond)+"-"+''.join(random.SystemRandom().choice(string.ascii_uppercase + string.digits) for _ in range(5))
training_data_file = open(file_path, "w+")

print("Vanilla MCTS")
play_game(game_setting) #First, we play a vanilla MCTS game

print("Bad vs good MCTS")
play_game(game_setting, bad_mcts=True) #We play one bad vs one good mcts against each other, only saving the good data


policies = [Policy(game_setting), Policy(game_setting)]
policies[0].import_data_and_train(25) #Training ANN with a maximum of 25 cases
policies[1].import_data_and_train() #Training ANN with all available training data
print("bad vs good ANN"), play_game(game_setting, policies, bad_vs_good_neural_net=True)


policies = [Policy(game_setting), Policy(game_setting)]
                                    #Note that the 0st neural net receives no training at all
policies[1].import_data_and_train() #Training ANN with all available training data
print("completely untrained ANN vs good ANN"), play_game(game_setting, policies, bad_vs_good_neural_net=True)


policies = [Policy(game_setting), Policy(game_setting)]
for policy in policies:
    policy.import_data_and_train()
print("good vs good ANN"), play_game(game_setting, policies=policies) #We play two ANNs against each other, saving the training data.


training_data_file.close() #Wrapping up the training data file.
#client = BSA()
#client = BSA.BasicClientActor.connect_to_server()
#print("yolo")
#client.connect()
#print("yolo")
#client.connect_to_server()

#client = BSA()
#client.connect_to_server()

M = tf.keras.models.load_model(ROOT_DIR + "model")
#model.compile(loss='mean_squared_error',
#            optimizer='Adam',
#            metrics=['accuracy'])


print("--- %s seconds ---" % (time.time() - start_time)) #How much time did we use?
