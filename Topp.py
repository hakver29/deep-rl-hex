from Policy import Policy
from prettytable import PrettyTable
from HexState import HexState1
from GameSetting import GameSetting
import random
import time
from definitions import MODEL_DIR
import time

class Topp:

    def __init__(self, game_setting):
        self.game_setting = game_setting
        self.game_setting.epochs = self.game_setting.topp_epochs
        self.K = game_setting.K
        self.G = game_setting.topp_G
        self.max_cases = game_setting.max_cases
        self.policies = []
        self.epsilon = game_setting.topp_epsilon
        self.negative_training_power = game_setting.negative_training_power

    def train_policies(self, load_best_policy=False, load_reinforcement=False):
        if load_reinforcement:
            for i in range(0, game_setting.K):
                policy = Policy(self.game_setting)
                nr_of_episodes = policy.load_reinforcement_model(i)
                self.policies.append([policy, nr_of_episodes, 0, 0])
            return

        if load_best_policy:
            start = 1
            policy = Policy(self.game_setting)
            nr_of_training_cases = policy.load_best_model()
            self.policies.append([policy, nr_of_training_cases, 0, 0])
        else:
            start = 0
        policy = Policy(self.game_setting)

        max_cases = min(policy.import_data_and_train(max_cases=self.max_cases), self.max_cases)

        if self.negative_training_power > 0:
            for i in range(start,self.K):
                policy = Policy(self.game_setting)

                nr_of_cases = max(1, max_cases//((i+1)**self.negative_training_power))

                actual_nr_of_cases = policy.import_data_and_train(max_cases=nr_of_cases)
                self.policies.append([policy, actual_nr_of_cases, 0,0])
        else:
            for i in range(start, self.K):
                policy = Policy(self.game_setting)
                nr_of_cases = max(int(max_cases * (self.K - i-1) / (self.K - 1)), 0)
                if nr_of_cases > 0:
                    actual_nr_of_cases = policy.import_data_and_train(max_cases=nr_of_cases)
                    self.policies.append([policy, actual_nr_of_cases, 0, 0])
                else:
                    self.policies.append([policy, 0, 0, 0])

    def play_tournament(self):
        for i in range(0,self.K-1):
            for k in range (i+1,self.K):
                print("\n" + str(i + 1) + ". best ANN vs. the " + str(k + 1) + " best ANN. Playing " + str(
                    self.G) + " games. Games are below.")
                for m in range(0, self.G):
                    result = self.play_game([self.policies[i], self.policies[k]],m)
                    if result == 0:
                        self.policies[i][2] += 1
                        self.policies[k][3] += 1
                    else:
                        self.policies[k][2] += 1
                        self.policies[i][3] += 1

        if self.game_setting.load_reinforcement:
            t = PrettyTable(['Theoretical rank', "Number of training episodes", 'Wins', 'Losses', '% Win'])
            for i in range(self.K-1, -1, -1):
                training_cases = self.policies[i][1]
                wins = self.policies[i][2]
                losses = self.policies[i][3]
                win_rate = (round(wins / (wins + losses), 2)) * 100
                t.add_row([self.K-i, training_cases, wins, losses, win_rate])
            print(t)
        else:
            t = PrettyTable(['Theoretical rank', "Number of training cases", 'Wins', 'Losses', '% Win'])
            for i in range(0, self.K):
                training_cases = self.policies[i][1]
                wins = self.policies[i][2]
                losses = self.policies[i][3]
                win_rate = (round(wins / (wins + losses), 2)) * 100
                t.add_row([i + 1, training_cases, wins, losses, win_rate])
            print(t)

        if self.most_trained_neural_net_has_most_wins() and self.game_setting.load_best_policy is False and self.game_setting.load_reinforcement is False:
            win_rate = (round(self.policies[0][2]/(self.policies[0][2]+self.policies[0][3]), 2))*100
            training_cases = self.policies[0][1]
            file_name = str(self.game_setting.network_dimensions[0])+"-"+str(win_rate)[0:4]+"-"+str(training_cases)+"-"+str(int(time.time()))
            self.policies[0][0].model.save(MODEL_DIR+file_name)
            print("Model was saved.")

    def play_game(self, policies, i):
        playerdict = {1: "white", 2: "black"}
        self.game_setting.P = playerdict[i%2+1]
        state = HexState1(self.game_setting)
        while (state.white_groups.connected(1, 2) == False and state.black_groups.connected(1, 2) == False):
            if self.game_setting.verbose >= 1:
                print(state)
            if self.game_setting.verbose >= 3:
                print(str(policies[state.toplay-1][1]) + " training cases network calculates.")
            random_num = random.uniform(0, 1)
            # If our random number exceeds epsilon, we let the ANN pick move. If not, the move is random.
            if random_num > self.epsilon:
                #def select(self, feature_vector, legal_moves, stochastic=False):
                policy = policies[state.toplay-1][0]
                legal_moves = [state.convertCoordinateToInteger(move) for move in state.moves()]
                feature_vector = state.convertFeatureVectorToFormat(state.board.flatten('F'))
                #print("Board representation sent to ANN: " + str(feature_vector))


                start_time = time.time()  # We start counting the time.
                integerMove = policy.select(feature_vector, legal_moves)

                if game_setting.verbose >= 3:
                    print("--- %s seconds ---" % (time.time() - start_time))  # How much time did we use?

                move = state.convertIntegerToCoordinate(integerMove)

                #print("ANN suggests moving " + str(move))
            else:
                move = random.choice(state.moves())
            if state.toplay == 2:
                state.place_black(move)
                state.set_turn(1)
            elif state.toplay == 1:
                state.place_white(move)
                state.set_turn(2)

            #print("\n\n")

        if self.game_setting.verbose >= 2:
            print(state)

        return state.winner()-1

    def most_trained_neural_net_has_most_wins(self):
        for i in range(1,self.K):
            if self.policies[0][2] <= self.policies[i][2]:
                return False
        return True


game_setting = GameSetting()
topp = Topp(game_setting)
topp.train_policies(load_best_policy=game_setting.load_best_policy, load_reinforcement=game_setting.load_reinforcement)
topp.play_tournament()