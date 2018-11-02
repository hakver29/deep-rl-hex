import numpy as np

class Board:
    def __init__(self, side):
        self.side = side
        self.map = [["0" for i in range(side)] for i in range(side)]

    def __repr__(self):
        """
        newrow = c.row + c.column  # (assume 0-based indexing for rows and columns)

        newcol = midcol + c.column - c.row

        (newrow,newcol) = coordinates in the diamond grid of cell c

        So DiamondGrid[newrow,newcol].value = c.value
        :return:
        """

        return str(self.map)