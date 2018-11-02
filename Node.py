from math import *
import random


class Node:
    """
    Node i treet. Resultatet til en node er alltid fra spilleren som nettopp beveget seg
    """

    def __init__(self, move=None, parent=None, state=None):
        self.move = move  # the move that got us to this node - "None" for the root node
        self.parentNode = parent  # "None" for the root node
        self.childNodes = []
        self.wins = 0
        self.visits = 0
        self.untried_moves = state.get_moves()  # child nodes
        self.player_just_moved = state.player_just_moved

    def select_child(self):
        """
        Heuristikk hentet fra https://en.wikipedia.org/wiki/Monte_Carlo_tree_search

        c.wins stands for the number of wins for the node considered after the i-th move
        c.visits stands for the number of simulations for the node considered after the i-th move
        self.visits stands for the total number of simulations after the i-th move
        """
        state = max(self.childNodes, key=lambda c: c.wins / c.visits + sqrt(2 * log(self.visits) / c.visits))
        return state

    def add_child(self, move, state):
        """
        Fjerner move fra untried_motves og lager en ny child node.
        Returnerer child-noden som er generert.
        """
        node = Node(move=move, parent=self, state=state)
        self.untried_moves.remove(move)
        self.childNodes.append(node)
        return node

    def update(self, result):
        """
        Oppdaterer antallet visits og wins på en node
        """
        self.visits += 1
        self.wins += result

    def children_to_string(self):
        """
        Brukes for å gi kontinuerlig statistikk om rollout-prosessen underveis i spillet.
        """
        s = ""
        for c in self.childNodes:
            s += str(c) + "\n"
        return s

    """
    Printer statistikken til en den gitte noden
    """

    def __repr__(self):
        return "[Move: " + str(self.move) + ", Wins/Visits: " + str(int(self.wins)) + "/" + str(int(self.visits)) + "]"