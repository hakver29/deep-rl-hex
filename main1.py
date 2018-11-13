#from Node import *
#from HexState import *
from Node1 import *
from HexState1 import *
from GameSetting import *
from Board import *
import copy

def tree_search(rootstate, itermax, verbose=False):
    rootnode = Node1(state=rootstate)

    for i in range(itermax):
        node = rootnode
        #state = rootstate.clone()
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
        while state.moves() != [] and state.winner() == 0:
            state.play(random.choice(state.moves()))

        # Backpropagation
        while node != None:
            node.visits += 1
            if rootstate.turn() == state.winner():
                node.wins += 1
            node = node.parentNode

    if game_setting.verbose == True:
        print(rootnode.children_to_string())
    return max(rootnode.childNodes, key=lambda c: c.visits).move


def play_game(game_setting):
    """
    Spiller et enkelt spill mellom to spillere
    """
    player_wins = {"black": 0, "white": 0}
    for i in range(game_setting.G):
        state = HexState1(game_setting)
        while (state.white_groups.connected(1,2) == False and state.black_groups.connected(1,2) == False):
            move = tree_search(rootstate=state, itermax=game_setting.M, verbose=game_setting.verbose)
            if state.toplay == 2:
                state.place_black(move)
                state.set_turn(1)
            elif state.toplay == 1:
                state.place_white(move)
                state.set_turn(2)

            print(state)

            if game_setting.verbose == True:
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

        if state.winner() == 1:
            player_wins["white"] += 1
        elif state.winner() == 2:
            player_wins["black"] += 1
    print(player_wins)

game_setting = GameSetting()
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

