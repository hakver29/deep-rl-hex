import numpy as np
from unionfind import unionfind

class HexState1:
    # Representerer en state av brettet i Hex

    # Oversikt av spillere
    PLAYERS = {"none" : 0, "white" : 1, "black" : 2}

    # Representasjon av kantene i spillet
    # Brukes i UnionFind
    EDGE1 = 1
    EDGE2 = 2

    neighbor_patterns = ((-1,0), (0,-1), (-1,1), (0,1), (1,0), (1,-1))

    def __init__(self, gamesetting, keith_state = None):
        # Initialiserer HexState
        self.size = gamesetting.size
        if keith_state == None:
            self.toplay = self.PLAYERS[gamesetting.P]
            self.board = np.zeros((gamesetting.size, gamesetting.size))
        else:
            keith_state = list(keith_state)
            self.board = np.reshape(keith_state[1:26], (5, 5))
            players = {1: "white", 2: "black"}
            self.toplay = self.PLAYERS[players[keith_state[0]]]
        self.gamesetting = gamesetting
        self.white_groups = unionfind()
        self.black_groups = unionfind()

    def play(self, cell):
        # Plasserer en brikke
        if(self.toplay == self.PLAYERS["white"]):
            self.place_white(cell)
            self.set_turn(2)
        elif(self.toplay == self.PLAYERS["black"]):
            self.place_black(cell)
            self.set_turn(1)

    def place_white(self, cell):
        # Plasserer en brikke av type Player 1
        if(self.board[cell] == self.PLAYERS["none"]):
            self.board[cell] = self.PLAYERS["white"]
        else:
            raise ValueError("Cell occupied")

        if(cell[0] == 0):
            self.white_groups.join(self.EDGE1, cell)
        if(cell[0] == self.size -1):
            self.white_groups.join(self.EDGE2, cell)

        for n in self.neighbors(cell):
            if(self.board[n] == self.PLAYERS["white"]):
                self.white_groups.join(n, cell)

    def place_black(self, cell):
        # Plasserer en brikke av type Player 2
        if(self.board[cell] == self.PLAYERS["none"]):
            self.board[cell] = self.PLAYERS["black"]
        else:
            raise ValueError("Cell occupied")

        if(cell[1] == 0):
            self.black_groups.join(self.EDGE1, cell)
        if(cell[1] == self.size -1):
            self.black_groups.join(self.EDGE2, cell)

        for n in self.neighbors(cell):
            if(self.board[n] == self.PLAYERS["black"]):
                self.black_groups.join(n, cell)

    def set_turn(self, player):
        # Setter hvem det er sin tur
        if(player in self.PLAYERS.values() and player !=self.PLAYERS["none"]):
            self.toplay = player
        else:
            raise ValueError('Invalid turn: ' + str(player))

    def winner(self):
        # Returnerer vinneren av spillet (1/2)
        # Returnerer 0 hvis ingen har vunnet
        if(self.white_groups.connected(self.EDGE1, self.EDGE2)):
            return self.PLAYERS["white"]
        elif(self.black_groups.connected(self.EDGE1, self.EDGE2)):
            return self.PLAYERS["black"]
        else:
            return self.PLAYERS["none"]

    def get_result(self, playertp):
        # Returnerer resultatet fra playerjm sitt ståsted
        if self.toplay == playertp:
            return 1.0
        else:
            return 0.0

    def neighbors(self, cell):
        # Returnerer naboene til cell
        x = cell[0]
        y = cell[1]
        return [(n[0]+x , n[1]+y) for n in self.neighbor_patterns\
            if (0<=n[0]+x and n[0]+x<self.size and 0<=n[1]+y and n[1]+y<self.size)]

    def moves(self):
        # Returnerer tilgjengelige moves til en state
        moves = []
        for y in range(self.size):
            for x in range(self.size):
                if self.board[x,y] == self.PLAYERS["none"]:
                    moves.append((x,y))
        return moves

    def convertIntegerToCoordinate(self, intMove):
        # Hjelpefunksjon
        ycoordinate = intMove // self.size
        xcoordinate = intMove % self.size
        return xcoordinate, ycoordinate

    def convertCoordinateToInteger(self, move):
        # Hjelpefunksjon
        return move[1] * self.size + move[0]

    def convertFeatureVectorToFormat(self, feature_vector):
        # Hjelpefunksjon
        feature_vector = feature_vector.tolist()
        feature_vector.insert(0,self.toplay)
        feature_vector = np.array(feature_vector)
        return feature_vector

    def __str__(self):
        # Printer representasjon av brettet
        white = 'O'
        black = '@'
        empty = '.'
        ret = '\n'
        coord_size = len(str(self.size))
        offset = 1
        ret+=' '*(offset+1)
        for x in range(self.size):
            ret+=chr(ord('A')+x)+' '*offset*2
        ret+='\n'
        for y in range(self.size):
            ret+=str(y+1)+' '*(offset*2+coord_size-len(str(y+1)))
            for x in range(self.size):
                if(self.board[x, y] == self.PLAYERS["white"]):
                    ret+=white
                elif(self.board[x,y] == self.PLAYERS["black"]):
                    ret+=black
                else:
                    ret+=empty
                ret+=' '*offset*2
            ret+=white+"\n"+' '*offset*(y+1)
        ret+=' '*(offset*2+1)+(black+' '*offset*2)*self.size

        return ret

def convertFeatureVectorToFormat(feature_vector, toplay):
    # Hjelpefunksjon
    feature_vector = feature_vector.tolist()
    feature_vector.insert(0,toplay)
    feature_vector = np.array(feature_vector)
    return feature_vector