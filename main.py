import copy

from GameSetting import *
from HexState import *
from HexState import convertFeatureVectorToFormat
from Node import *
from Policy import *
from definitions import REINFORCEMENT_MODEL_DIR
import requests
import socket

def get_save_model_indices():
    indices = []
    for i in range(0, game_setting.K-1):
        factor = i/(game_setting.K-1)
        indices.append(int(factor * game_setting.G))
    indices.append(game_setting.G)
    return indices

def tree_search(rootstate, itermax, verbose=False, policy=None, policies=None, save_training=True, moves_are_random=False):
    rootnode = Node1(state=rootstate)
    if game_setting.verbose >= 1:
        print("Starting " + str(itermax) + " rollouts.")
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
                    flattened_move = policies[state.toplay-1].select(state.convertFeatureVectorToFormat(rootstate.board.flatten('F')),
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

    if game_setting.verbose >= 2:
        print(rootnode.children_to_string())

    feature_vector, target = get_feature_and_target(rootnode, rootstate, rootstate.toplay)
    replay_buffer.append([feature_vector, target])

    if save_training:
        append_result_to_training_data(feature_vector, target, itermax, moves_are_random=moves_are_random)

    if moves_are_random:
        return random.choice(rootnode.childNodes).move
    else:
        return max(rootnode.childNodes, key=lambda c: c.visits).move


def play_game(game_setting, game_func, policies=None):
    if policies[0] is not None and policies[0] is policies[1]:
        reinforcement_learning = True
        save_model_at_indices = get_save_model_indices()
    else:
        reinforcement_learning = False

    player_wins = {"black": 0, "white": 0}
    for i in range(game_setting.G):
        playerdict = {1:"white", 2: "black"}
        game_setting.P = playerdict[i%2 + 1]
        state = HexState1(game_setting)

        if reinforcement_learning and i in save_model_at_indices:
            file_name = str(game_setting.network_dimensions[0]) + "-NA-" + str(i) + "-" + str(int(time.time()))
            policies[0].model.save(REINFORCEMENT_MODEL_DIR + file_name)

        while state.winner() == 0:
            move = game_func(state, policies)
            if state.toplay == 2:
                state.place_black(move)
                state.set_turn(1)
            elif state.toplay == 1:
                state.place_white(move)
                state.set_turn(2)

            if game_setting.verbose >= 2:
                print(state)
                if state.toplay == 1:
                    print("Player 2 selects " + str(move) + "\n")
                elif state.toplay == 2:
                    print("Player 1 selects " + str(move) + "\n")
                #print("Player " + str(state.toplay) + " selects " + str(move))
        if game_setting.verbose >= 1:
            if (i % 5) == 0:
                print("We have played " + str(i) + " games. player_wins: " + str(player_wins))
            print(state)
            if state.winner() == 2:
                print("\nPlayer black wins")
            elif state.winner() == 1:
                print("\nPlayer white wins")

        if state.winner() == 1:
            player_wins["white"] += 1
        elif state.winner() == 2:
            player_wins["black"] += 1

        if reinforcement_learning:
            random_rbuf = random.sample(replay_buffer, random.randint(1, len(replay_buffer)-1))
            for case in random_rbuf:
                feature_vector = np.array([np.array(case[0])])
                target_vector = np.array([np.array(case[1])])
                policies[0].model.fit(feature_vector, target_vector)
            if game_setting.verbose >= 2:
                policies[0].model.summary()

        del replay_buffer[:]

    if reinforcement_learning:
        file_name = str(game_setting.network_dimensions[0]) + "-NA-" + str(game_setting.G) + "-" + str(int(time.time()))
        policies[0].model.save(REINFORCEMENT_MODEL_DIR + file_name)

    print(player_wins)


def get_feature_and_target(rootnode, rootstate, toplay):
    target = [0] * game_setting.nr_of_legal_moves

    for child_node in rootnode.childNodes:
        move = child_node.move[1] * game_setting.size + child_node.move[0]
        target[move] = child_node.visits / rootnode.visits

    feature_vector = convertFeatureVectorToFormat(rootstate.board.flatten('F'), toplay)

    return feature_vector, target

def append_result_to_training_data(feature_vector, target, itermax,moves_are_random=False):
    if moves_are_random:
        line = ",".join(str(int(input)) for input in feature_vector) + "|" + ",".join(
            str(target) for target in target) + "|" + str(itermax) + "|" + "random move\n"
    else:
        line = ",".join(str(int(input)) for input in feature_vector)+"|"+",".join(str(target) for target in target)+"|"+str(itermax)+"\n"

    #training_data_file.write(line)
    line = line[:-1]
    encoded_line = (line+socket.gethostname()).encode()

    headers = {'Content-Length2': str(len(encoded_line))}
    r = requests.post("http://sikkerhetshull.no:12001", data={(line+socket.gethostname()).encode()}, headers=headers)
    print(r)


def calculate_itermax(state):
    remaining_moves = len(state.moves())
    if remaining_moves < 10:
        return 2000
    if remaining_moves < 17:
        return 10000
    if remaining_moves < 26:
        return 20000
    if remaining_moves < 37:
        return 100000
    return game_setting.M



def bad_mcts(state, policies):
    moves_are_random = False

    genius_player = random.randint(1, 2)
    if state.toplay == genius_player:
        itermax = calculate_itermax(state)
        move = tree_search(rootstate=state, itermax=itermax, verbose=game_setting.verbose,
                           policies=None, save_training=True, moves_are_random=moves_are_random)
    elif state.toplay == 3 - genius_player:
        move = tree_search(rootstate=state, itermax=1, verbose=game_setting.verbose,
                           policies=None, save_training=False, moves_are_random=moves_are_random)

    return move

def random_mcts(state, policies):
    moves_are_random = True
    itermax = calculate_itermax(state)
    return tree_search(rootstate=state, itermax=itermax, verbose=game_setting.verbose,
                       moves_are_random=moves_are_random, policies=None)

def pure_mcts(state, policies):
    moves_are_random = False
    itermax = calculate_itermax(state)
    return tree_search(rootstate=state, itermax=itermax, verbose=game_setting.verbose,
                moves_are_random=moves_are_random, policies=None)

def neural_net_vs_neural_net(state, policies):
    moves_are_random = False

    if state.toplay == 2:
        move = tree_search(rootstate=state, itermax=game_setting.M, verbose=game_setting.verbose,
                           policies=policies, save_training=True, moves_are_random=moves_are_random)
    elif state.toplay == 1:
        move = tree_search(rootstate=state, itermax=game_setting.M, verbose=game_setting.verbose,
                           policies=policies, save_training=False, moves_are_random=moves_are_random)

    return move

def train_neural_net_by_reinforcement(state, policies):
    moves_are_random = False

    return tree_search(rootstate=state, itermax=game_setting.M, verbose=game_setting.verbose,
                       moves_are_random=moves_are_random, policies=policies, save_training=False)

def play_bad_mcts():
    print("Bad vs good MCTS")
    #Bad mcts mode pits a strong mcts player versus a weak mcts player. The idea is make
    #training data that teaches ANNs how to not only play vs strong players, but weak
    #players too. Weak players make random moves in odd corners of the board that the
    #ANN may respond erratically to.
    policies = [None, None]
    play_game(game_setting, bad_mcts, policies=policies)

def play_random_mcts():
    print("Random move MCTS")
    #Random MCTS mode makes all moves be random, but each board position is rigorously
    #evaluated by MCTS with plenty of simulations.
    policies = [None, None]
    play_game(game_setting, random_mcts, policies=policies )

def play_pure_mcts():
    print("Vanilla MCTS")
    # Pure MCTS pits two strong MCTS players against each other.
    policies = [None, None]
    play_game(game_setting, pure_mcts, policies=policies)


def play_good_vs_bad_neural_net():
    print("good vs bad neural net")
    policies = [Policy(game_setting), Policy(game_setting)]
    policies[0].import_data_and_train(25)  # Training ANN with a maximum of 25 cases
    policies[1].import_data_and_train()  # Training ANN with all available training data
    play_game(game_setting, neural_net_vs_neural_net, policies=policies)

def play_good_vs_good_neural_net():
    print("Good vs good neural net")
    policies = [Policy(game_setting), Policy(game_setting)]
    policies[0].import_data_and_train()  # Training ANN with all available training data
    policies[1].import_data_and_train()  # Training ANN with all available training data
    play_game(game_setting, neural_net_vs_neural_net, policies=policies)

def play_reinforcement_neural_net():
    print("Reinforcement learning")
    policy = Policy(game_setting)
    policies = [policy, policy]
    play_game(game_setting, neural_net_vs_neural_net, policies=policies)


start_time = time.time() #We start counting the time.

game_setting = GameSetting() #Load the game settings.

replay_buffer = []

file_path = training_data_file_path = DATA_DIR+'n'.join(str(dim) for dim in game_setting.network_dimensions)+"-"+str(time.time()+datetime.now().microsecond)+"-"+''.join(random.SystemRandom().choice(string.ascii_uppercase + string.digits) for _ in range(5))
training_data_file = open(file_path, "w+")

time_end = 1543478400

while time.time() < time_end:
    print("Script will run until unixtime " + str(time_end) + ". Hours left: " + str((time_end-time.time())/3600))
    play_pure_mcts()
    play_bad_mcts()
    play_random_mcts()

#play_good_vs_bad_neural_net()
#play_good_vs_good_neural_net()
#play_reinforcement_neural_net()


training_data_file.close() #Wrapping up the training data file.

print("--- %s seconds ---" % (time.time() - start_time)) #How much time did we use?
