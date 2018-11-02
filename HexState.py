import numpy as np

class HexState:
    # Representerer en tilstand av Nim. NimState tar inn spillets regler og antall steiner som er gjenværende.
    PLAYERS = {"none": 0, "white": 1, "black": 2}


    def __init__(self, game_setting):
        self.size = game_setting.size
        self.player_just_moved = game_setting.P
        self.board = np.zeros((game_setting.size,game_setting.size))
        self.game_setting = game_setting
        self.toplay = self.PLAYERS["white"]


    def clone(self):
        """
        Lager en deep clone av game state
        """
        st = HexState(self.game_setting)
        st.player_just_moved = self.player_just_moved
        return st

    def do_move(self, move):
        """
        Oppdaterer spillet ved å utføre move
        player_just_moved oppdaterer seg deretter
        """

        self.player_just_moved = 3 - self.player_just_moved

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

    def neigbors(self, cell):
        x = cell[0]
        y = cell[1]
        return [(n[0] + x, n[1] + y) for n in self.neighbor_patterns \
                if (0 <= n[0] + x and n[0] + x < self.size and 0 <= n[1] + y and n[1] + y < self.size)]

    def get_result(self, playerjm):
        """
        Returnerer resultatet fra playerjm sitt ståsted
        """
        assert self.stones_remaining == 0
        if self.player_just_moved == playerjm:
            return 1.0
        else:
            return 0.0

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