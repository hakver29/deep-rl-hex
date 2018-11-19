#from Node import *
#from HexState import *
from Node import *
from HexState import *
from GameSetting import *
import copy
from Policy import Policy


def convertIntegerToCoordinate(intMove, boardSize):
    ycoordinate = intMove//boardSize
    xcoordinate = intMove%boardSize
    return xcoordinate,ycoordinate


def convertCoordinateToInteger(move, boardSize):
    return move[1]*boardSize + move[0]



def tree_search(rootstate, itermax, verbose=False, policy=None, debug=False):
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

        #print(" ")

        if debug == True:
            print("Starting rollout from state below, after completing selection phase.")
            print(state)

        #print(" ")

        # Expansion
        if node.untried_moves != []:
            move = random.choice(node.untried_moves)
            state.play(move)
            node = node.add_child(move, state)

        if debug == True:
            print("Randomly chose " + str(move) + " among the untried moves, in expansion phase.")
            print(state)

        #print(" ")

        # Simulation
        while state.winner() == 0:

            random_move = random.choice(state.moves())
            state.play(random.choice(state.moves()))
            if debug == True:
                print("Player " + str(game_setting.player_symbols[state.toplay-1]) + " just played " + str(random_move) + ".")
                print(state)
                print(" ")

        winner = {1: "white", 2: "black"}

        if debug is True:
            print("Board above is a win for " + winner[state.winner()])
            print(" ")

        # Backpropagation
        while node != None:
            node.visits += 1
            if node.toplay == state.winner():
                node.wins += 1
                if debug is True:
                    if node.move is not None:
                        print("Adding win to node with move " + str(node.move) + ". Current statistic is " + str(node.wins) + "/" + str(node.visits))
                    else:
                        print("Adding win to rootnode. Current statistic is " + str(node.wins) + "/" + str(node.visits))
                    #print(" ")
            node = node.parentNode

    if game_setting.verbose == True:
        print(rootnode.children_to_string())

    return max(rootnode.childNodes, key=lambda c: c.visits).move
    if policy is None and training_data_file is not None:
        append_mcts_result_to_training_data(rootnode, rootstate)
        return max(rootnode.childNodes, key=lambda c: c.visits).move
    elif policy is not None:
        intMove = policy.select(rootstate.board.flatten(), [convertCoordinateToInteger(childNode.move, game_setting.size) for childNode in rootnode.childNodes])
        return convertIntegerToCoordinate(intMove,game_setting.size)


def play_game(game_setting, policy=None):
    """
    Spiller et enkelt spill mellom to spillere
    """
    player_wins = {"black": 0, "white": 0}
    for i in range(game_setting.G):
        state = HexState1(game_setting)
        while (state.white_groups.connected(1,2) == False and state.black_groups.connected(1,2) == False):
            move = tree_search(rootstate=state, itermax=game_setting.M, verbose=game_setting.verbose, policy=policy)
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


def append_mcts_result_to_training_data(rootnode, rootstate):
    target = [0] * game_setting.size**2

    for child_node in rootnode.childNodes:
        move = child_node.move[1]*game_setting.size+child_node.move[0]
        target[move] = child_node.wins/child_node.visits

    input = rootstate.board.flatten()
    for i in range(0,len(input)):
        if input[i] == float(rootstate.toplay):
            input[i] = 1
        elif input[i] != 0.0:
            input[i] = -1


    training_data_file.write(",".join(str(int(input)) for input in input)+"|"+",".join(str(target) for target in target)+"|"+"\n")



game_setting = GameSetting()
#file_path = training_data_file_path = DATA_DIR+'n'.join(str(dim) for dim in game_setting.network_dimensions)+"-"+str(time.time()+datetime.now().microsecond)+"-"+''.join(random.SystemRandom().choice(string.ascii_uppercase + string.digits) for _ in range(5))
training_data_file = None
#training_data_file = open(file_path, "w+")
"""
state = HexState1(game_setting)
print(state)
print(state.place_white((1,1)))
print(state.place_black((0,0)))
print(state.place_white((1,0)))
print(state.place_black((0,1)))
print(state)
print(state.winner())
"""
play_game(game_setting)
#training_data_file.close()
#policy = Policy(game_setting)
#policy.import_all_data_and_train()
#play_game(game_setting,policy=policy)


