from Policy import Policy
from prettytable import PrettyTable
from HexState import HexState1
from GameSetting import GameSetting
import random

class Topp:

    def __init__(self, game_setting):
        self.game_setting = game_setting
        self.K = game_setting.K
        self.G = game_setting.topp_G
        self.max_cases = game_setting.max_cases
        self.policies = []
        self.epsilon = game_setting.topp_epsilon

    def train_policies(self):
        policy = Policy(self.game_setting)
        max_cases = min(policy.import_data_and_train(max_cases=self.max_cases), self.max_cases)

        for i in range(0,self.K):
            policy = Policy(self.game_setting)
            nr_of_cases = max_cases//(i+1)
            actual_nr_of_cases = policy.import_data_and_train(max_cases=nr_of_cases)
            self.policies.append([policy, actual_nr_of_cases, 0,0])

    def play_tournament(self):
        for i in range(0,self.K-1):
            for k in range (i+1,self.K):
                for m in range(0, self.G):
                    print("\n"+str(i) + ". best ANN vs. the " + str(k) + " best ANN. Playing " + str(self.G) + " games. Games are below.")
                    result = self.play_game([self.policies[i][0], self.policies[k][0]],m)
                    if result == 0:
                        self.policies[i][2] += 1
                        self.policies[k][3] += 1
                    else:
                        self.policies[k][2] += 1
                        self.policies[i][3] += 1

        t = PrettyTable(['Number of training cases', 'Wins', 'Losses'])

        for i in range(0,self.K):
            t.add_row([str(self.policies[i][1]), str(self.policies[i][2]), str(self.policies[i][3])])
        print(t)

    def play_game(self, policies, i):
        if i%2 + 1 == 1:
            self.game_setting.P = "white"
        else:
            self.game_setting.P = "black"
        state = HexState1(self.game_setting)
        while (state.white_groups.connected(1, 2) == False and state.black_groups.connected(1, 2) == False):
            random_num = random.uniform(0, 1)
            # If our random number exceeds epsilon, we let the ANN pick move. If not, the move is random.
            if random_num > self.epsilon:
                #def select(self, feature_vector, legal_moves, stochastic=False):
                policy = policies[state.toplay-1]
                legal_moves = [state.convertCoordinateToInteger(move) for move in state.moves()]
                integerMove = policy.select(state.convertFeatureVectorToFormat(state.board.flatten('F'), state.toplay), legal_moves)
                move = state.convertIntegerToCoordinate(integerMove)
            else:
                move = random.choice(state.moves())
            if state.toplay == 2:
                state.place_black(move)
                state.set_turn(1)
            elif state.toplay == 1:
                state.place_white(move)
                state.set_turn(2)
        players = {1: "white", 2: "black"}
        print(state)
        print(players[state.winner()] + " wins.")
        return state.winner()-1

game_setting = GameSetting()
topp = Topp(game_setting)
topp.train_policies()
topp.play_tournament()