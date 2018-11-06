import numpy as np
from unionfind import *

class HexState:
    # Representerer en tilstand av Nim. NimState tar inn spillets regler og antall steiner som er gjenværende.
    PLAYERS = {"none": 0, "white": 1, "black": 2}
    EDGE1 = 1
    EDGE2 = 2
    GAMEOVER = -1
    neighbor_patterns = ((-1, 0), (0, -1), (-1, 1), (0, 1), (1, 0), (1, -1))


    def __init__(self, game_setting):
        self.size = game_setting.size
        self.player_just_moved = game_setting.P
        self.board = np.zeros((game_setting.size,game_setting.size))
        self.game_setting = game_setting

        self.white = unionfind()
        self.black = unionfind()

    def play(self, cell):
        """
        Play a stone of the current turns color in the passed cell.
        """
        if (self.toplay == self.PLAYERS["white"]):
            self.place_white(cell)
            self.toplay = self.PLAYERS["black"]
        elif (self.toplay == self.PLAYERS["black"]):
            self.place_black(cell)
            self.toplay = self.PLAYERS["white"]

    def do_move(self, cell, player_just_moved):
        """
        Place a white stone regardless of whose turn it is.
        """
        if (self.board[cell] == self.PLAYERS["none"]):
            self.board[cell] = self.PLAYERS[player_just_moved]
        else:
            raise ValueError("Cell occupied")
        # if the placed cell touches a white edge connect it appropriately
        if (cell[0] == 0):
            self.groups.join(self.EDGE1, cell)
        if (cell[0] == self.size - 1):
            self.groups.join(self.EDGE2, cell)
        # join any groups connected by the new white stone
        for n in self.neighbors(cell):
            if (self.board[n] == self.PLAYERS["white"]):
                self.white.join(n, cell)
            elif (self.board[n] == self.PLAYERS["black"]):
                self.black.join(n, cell)

    def place_white(self, cell):
        """
        Place a white stone regardless of whose turn it is.
        """
        if (self.board[cell] == self.PLAYERS["none"]):
            self.board[cell] = self.PLAYERS["white"]
        else:
            raise ValueError("Cell occupied")
        # if the placed cell touches a white edge connect it appropriately
        if (cell[0] == 0):
            self.white.join(self.EDGE1, cell)
        if (cell[0] == self.size - 1):
            self.white.join(self.EDGE2, cell)
        # join any groups connected by the new white stone
        for n in self.neighbors(cell):
            if (self.board[n] == self.PLAYERS["white"]):
                self.white.join(n, cell)

    def place_black(self, cell):
        """
        Place a black stone regardless of whose turn it is.
        """
        if (self.board[cell] == self.PLAYERS["none"]):
            self.board[cell] = self.PLAYERS["black"]
        else:
            raise ValueError("Cell occupied")
        # if the placed cell touches a black edge connect it appropriately
        if (cell[1] == 0):
            self.black.join(self.EDGE1, cell)
        if (cell[1] == self.size - 1):
            self.black.join(self.EDGE2, cell)
        # join any groups connected by the new black stone
        for n in self.neighbors(cell):
            if (self.board[n] == self.PLAYERS["black"]):
                self.black.join(n, cell)

    def clone(self):
        """
        Lager en deep clone av game state
        """
        st = HexState(self.game_setting)
        st.player_just_moved = self.player_just_moved
        return st

    def get_moves(self):
        """
        Returnerer alle tilgjengelige moves
        """
        moves = []
        for y in range(self.size):
            for x in range(self.size):
                if self.board[x, y] == self.PLAYERS["none"]:
                    moves.append((x, y))
        return moves

    def neighbors(self, cell):
        x = cell[0]
        y = cell[1]
        return [(n[0] + x, n[1] + y) for n in self.neighbor_patterns \
                if (0 <= n[0] + x and n[0] + x < self.size and 0 <= n[1] + y and n[1] + y < self.size)]

    def get_result(self):
        if self.black.connected == True:
                return "black"
        elif self.white.connected == True:
                return "white"

    def __str__(self):
        """
        Print an ascii representation of the game board.
        """
        white = 'O'
        black = '@'
        empty = '.'
        ret = '\n'
        coord_size = len(str(self.size))
        offset = 1
        ret += ' ' * (offset + 1)
        for x in range(self.size):
            ret += chr(ord('A') + x) + ' ' * offset * 2
        ret += '\n'
        for y in range(self.size):
            ret += str(y + 1) + ' ' * (offset * 2 + coord_size - len(str(y + 1)))
            for x in range(self.size):
                if (self.board[x, y] == self.PLAYERS["white"]):
                    ret += white
                elif (self.board[x, y] == self.PLAYERS["black"]):
                    ret += black
                else:
                    ret += empty
                ret += ' ' * offset * 2
            ret += white + "\n" + ' ' * offset * (y + 1)
        ret += ' ' * (offset * 2 + 1) + (black + ' ' * offset * 2) * self.size

        return ret