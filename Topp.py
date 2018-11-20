from Policy import Policy
from prettytable import PrettyTable
from HexState import HexState1

class Topp:

    def __init__(self, game_setting):
        self.game_setting = game_setting
        self.K = game_setting.K
        self.G = game_setting.topp_G
        self.max_cases = game_setting.max_cases
        self.policies = []

    def train_policies(self):
        for i in range(0,self.K):
            policy = Policy(self.game_setting)
            nr_of_cases = self.max_cases//(i+1)
            actual_nr_of_cases = policy.import_data_and_train(max_cases=nr_of_cases)
            self.policies.append([policy, actual_nr_of_cases, 0,0])

    def play_tournament(self):
        for i in range(0,self.K-1):
            for k in range (i+1,self.K):
                for m in range(0, self.G):
                    result = self.play_game([self.policies[i][0], self.policies[k][0]])
                    if result == 0:
                        self.policies[i][2] += 1
                    else:
                        self.policies[k][2] += 1



    def play_game(self, policies):
        state = HexState1(self.game_setting)
        while (state.white_groups.connected(1, 2) == False and state.black_groups.connected(1, 2) == False):
            #def select(self, feature_vector, legal_moves, stochastic=False):
            move = policies[state.toplay-1][0].select(state.board.flatten('F'))
            if state.toplay == 2:
                state.place_black(move)
                state.set_turn(1)
            elif state.toplay == 1:
                state.place_white(move)
                state.set_turn(2)
