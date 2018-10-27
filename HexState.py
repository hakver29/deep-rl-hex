class NimState:
    """ A state of the game Nim. In Nim, players alternately take 1,2 or 3 chips with the
        winner being the player to take the last chip.
        In Nim any initial state of the form 4n+k for k = 1,2,3 is a win for player 1
        (by choosing k) chips.
        Any initial state of the form 4n is a win for player 2.
    """

    def __init__(self, game_setting, stones):
        # self.playerJustMoved = 2  # At the root pretend the player just moved is p2 - p1 has the first move
        # self.chips = ch
        self.stones_remaining = stones
        self.playerJustMoved = game_setting.P
        self.game_setting = game_setting

    def Clone(self):
        """ Create a deep clone of this game state.
        """
        st = NimState(self.game_setting, self.stones_remaining)
        st.playerJustMoved = self.playerJustMoved
        return st

    def DoMove(self, move):
        """ Update a state by carrying out the given move.
            Must update playerJustMoved.
        """
        # assert move >= 1 and move <= 3 and move == int(move)
        self.stones_remaining -= move
        self.playerJustMoved = 3 - self.playerJustMoved

    def get_moves(self):
        """ Get all possible moves from this state.
        """
        K = self.game_setting.K

        if K > self.stones_remaining:
            return [i for i in range(1, self.stones_remaining + 1)]
        else:
            return [i for i in range(1, K + 1)]

    def GetResult(self, playerjm):
        """ Get the game result from the viewpoint of playerjm.
        """
        assert self.stones_remaining == 0
        if self.playerJustMoved == playerjm:
            return 1.0  # playerjm took the last chip and has won
        else:
            return 0.0  # playerjm's opponent took the last chip and has won