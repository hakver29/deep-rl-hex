from Node import *
from HexState import *
from GameSetting import *
from Board import *

def tree_search(rootstate, itermax, verbose=False):
    rootnode = Node(state=rootstate)

    for i in range(itermax):
        node = rootnode
        state = rootstate.clone()

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
            state.do_move(node.move)

        # Expansion
        if node.untried_moves != []:
            move = random.choice(node.untried_moves)
            state.do_move(move)
            node = node.add_child(move, state)

        # Simulation
        while state.get_moves() != []:
            state.do_move(random.choice(state.get_moves()))

        # Backpropagation
        while node != None:
            node.update(state.get_result(node.player_just_moved))
            node = node.parentNode

    if game_setting.verbose == True:
        print(rootnode.children_to_string())
    return max(rootnode.childNodes, key=lambda c: c.visits).move


def play_game(game_setting):
    """
    Spiller et enkelt spill mellom to spillere
    """
    player_wins = [0,0]
    for i in range(game_setting.G):
        state = HexState(game_setting)
        while (state.white.connected(1,2) == False and state.black.connected(1,2) == False):
            move = tree_search(rootstate=state, itermax=game_setting.M, verbose=game_setting.verbose)
            state.do_move(move)
            if game_setting.verbose == True:
                if move == 1:
                    print("Player " + str(state.player_just_moved) + " selects " + str(
                        move))
                else:
                    print("Player " + str(state.player_just_moved) + " selects " + str(
                        move))

        if game_setting.verbose == True:
            if state.get_result(state.player_just_moved) == 1.0:
                print("Player " + str(state.player_just_moved) + " wins" + "\n")
            elif state.get_result(state.player_just_moved) == 0.0:
                print("Player " + str(3 - state.player_just_moved) + " wins" + "\n")

        player_wins[state.player_just_moved - 1] += 1
        print(state)

    print(player_wins)


#game_setting = GameSetting()
#play_game(game_setting)

game_setting = GameSetting()
play_game(game_setting)

"""
state = HexState(game_setting)
state.place_black((0,0))
state.place_black((0,2))
state.place_black((0,1))
#state.place_white((0,1))
state.place_white((1,1))
state.place_white((2,1))
print(state)
print(state.white.connected(state.EDGE1,state.EDGE2))
print(state.black.connected(state.EDGE1,state.EDGE2))
"""

"WHITE: 1"
"BLACK: 2"