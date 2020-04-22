import collections
import copy
import datetime
import math
import os
import pickle
import random

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset
from tqdm import tqdm


def save_as_pickle(filename, data):
    completeName = os.path.join("./datasets/",
                                filename)
    with open(completeName, 'wb') as output:
        pickle.dump(data, output)


def load_pickle(filename):
    completeName = os.path.join("./datasets/",
                                filename)
    with open(completeName, 'rb') as pkl_file:
        data = pickle.load(pkl_file)
    return data


def max_element_repeated_in_row(iterable):
    d = {}
    e = None
    c = 0
    for i, element in enumerate(iterable):
        if e is None:
            c = 1
            e = element
        elif e != element:
            counted = d.get(e, None)
            if counted is None:
                d[e] = c
            else:
                if c > counted:
                    d[e] = c
            c = 1
            e = element
        else:
            c += 1
            if i + 1 == len(iterable):
                counted = d.get(e, None)
                if counted is None:
                    d[e] = c
                else:
                    if c > counted:
                        d[e] = c
    return d


def prev_current(iterable):
    prv = None
    try:
        cur = next(iterable)
        while True:
            yield prv, cur
            prv = cur
            cur = next(iterable)
    except StopIteration:
        pass


class Board(object):
    def __init__(self):
        self.board = np.zeros([6, 7], dtype=np.int8)
        self.player = 1

    def move(self, column):
        if self.board[0, column] != 0:
            return "Invalid move"
        placed_row = None
        for i, (prev_row, current_row) in enumerate(
                prev_current(self.board.__iter__())):
            if current_row[column] != 0:
                prev_row[column] = self.player
                placed_row = i - 1
                break
        else:
            self.board[-1, column] = self.player
            placed_row = len(self.board) - 1
        won = self.check_if_won(self.player, placed_row, column)
        self.player = 2 if self.player == 1 else 1
        return won

    def check_if_won(self, player, placed_row, placed_column):
        row_results = max_element_repeated_in_row(self.board[placed_row])
        if row_results.get(player, 0) >= 4:
            return True
        column_result = max_element_repeated_in_row(
            self.board[:, placed_column].T)
        if column_result.get(player, 0) >= 4:
            return True
        major = np.diagonal(self.board, offset=(placed_column - placed_row))
        major_result = max_element_repeated_in_row(major)
        if major_result.get(player, 0) >= 4:
            return True
        minor = np.diagonal(np.rot90(self.board),
                            offset=-self.board.shape[1] + (
                                        placed_column + placed_row) + 1)
        minor_result = max_element_repeated_in_row(minor)
        if minor_result.get(player, 0) >= 4:
            return True
        return False

    def possible_actions(self):
        return [i for i, x in enumerate(self.board[0]) if x == 0]


# Encoder to encode Connect4 board for neural net input
def encode_board(board: Board):
    board_state = board.board
    encoded = np.zeros([6, 7, 3]).astype(int)
    encoder_dict = {1: 0, 2: 1}
    for row in range(6):
        for col in range(7):
            if board_state[row, col] != 0:
                encoded[row, col, encoder_dict[board_state[row, col]]] = 1
    if board.player == 1:
        encoded[:, :, 2] = 1  # player to move
    return encoded


class ConvBlock(nn.Module):
    def __init__(self):
        super(ConvBlock, self).__init__()
        self.action_size = 7
        self.conv1 = nn.Conv2d(3, 128, 3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(128)

    def forward(self, s):
        s = s.view(-1, 3, 6, 7)  # batch_size x channels x board_x x board_y
        s = F.relu(self.bn1(self.conv1(s)))
        return s


class ResBlock(nn.Module):
    def __init__(self, inplanes=128, planes=128, stride=1, downsample=None):
        super(ResBlock, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = F.relu(self.bn1(out))
        out = self.conv2(out)
        out = self.bn2(out)
        out += residual
        out = F.relu(out)
        return out


class OutBlock(nn.Module):
    def __init__(self):
        super(OutBlock, self).__init__()
        self.conv = nn.Conv2d(128, 3, kernel_size=1)  # value head
        self.bn = nn.BatchNorm2d(3)
        self.fc1 = nn.Linear(3 * 6 * 7, 32)
        self.fc2 = nn.Linear(32, 1)

        self.conv1 = nn.Conv2d(128, 32, kernel_size=1)  # policy head
        self.bn1 = nn.BatchNorm2d(32)
        self.logsoftmax = nn.LogSoftmax(dim=1)
        self.fc = nn.Linear(6 * 7 * 32, 7)

    def forward(self, s):
        v = F.relu(self.bn(self.conv(s)))  # value head
        v = v.view(-1, 3 * 6 * 7)  # batch_size X channel X height X width
        v = F.relu(self.fc1(v))
        v = torch.tanh(self.fc2(v))

        p = F.relu(self.bn1(self.conv1(s)))  # policy head
        p = p.view(-1, 6 * 7 * 32)
        p = self.fc(p)
        p = self.logsoftmax(p).exp()
        return p, v


class ConnectNet(nn.Module):
    def __init__(self):
        super(ConnectNet, self).__init__()
        self.conv = ConvBlock()
        for block in range(19):
            setattr(self, "res_%i" % block, ResBlock())
        self.outblock = OutBlock()

    def forward(self, s):
        s = self.conv(s)
        for block in range(19):
            s = getattr(self, "res_%i" % block)(s)
        s = self.outblock(s)
        return s


### Neural Net loss function implemented via PyTorch
class AlphaLoss(nn.Module):
    def __init__(self):
        super(AlphaLoss, self).__init__()

    def forward(self, y_value, value, y_policy, policy):
        value_error = (value - y_value) ** 2
        policy_error = torch.sum((-policy *
                                  (1e-8 + y_policy.float()).float().log()), 1)
        total_error = (value_error.view(-1).float() + policy_error).mean()
        return total_error


def select_leaf(self):
    current = self
    while current.is_expanded:
        best_move = current.best_child()
        current = current.maybe_add_child(best_move)
    return current


def expand(self, child_priors):
    self.is_expanded = True
    action_idxs = self.game.actions()
    c_p = child_priors
    if not action_idxs:
        self.is_expanded = False
    self.action_idxes = action_idxs
    # mask all illegal actions
    for i in range(len(child_priors)):
        if i not in action_idxs:
            c_p[i] = 0.0000000000
    # add dirichlet noise to child_priors in root node
    if self.parent.parent is None:
        c_p = self.add_dirichlet_noise(action_idxs, c_p)
    self.child_priors = c_p


def backup(self, value_estimate: float):
    current = self
    while current.parent is not None:
        current.number_visits += 1
        if current.game.player == 1:  # same as current.parent.game.player = 0
            current.total_value += (
                        1 * value_estimate)  # value estimate +1 = O wins
        elif current.game.player == 0:  # same as current.parent.game.player = 1
            current.total_value += (-1 * value_estimate)
        current = current.parent


class UCTNode(object):
    def __init__(self, game, move, parent=None):
        self.game = game  # state s
        self.move = move  # action index
        self.is_expanded = False
        self.parent = parent
        self.children = {}
        self.child_priors = np.zeros([7], dtype=np.float32)
        self.child_total_value = np.zeros([7], dtype=np.float32)
        self.child_number_visits = np.zeros([7], dtype=np.float32)
        self.action_idxes = []

    @property
    def number_visits(self):
        return self.parent.child_number_visits[self.move]

    @number_visits.setter
    def number_visits(self, value):
        self.parent.child_number_visits[self.move] = value

    @property
    def total_value(self):
        return self.parent.child_total_value[self.move]

    @total_value.setter
    def total_value(self, value):
        self.parent.child_total_value[self.move] = value

    def child_Q(self):
        return self.child_total_value / (1 + self.child_number_visits)

    def child_U(self):
        return math.sqrt(self.number_visits) * (
                abs(self.child_priors) / (1 + self.child_number_visits))

    def best_child(self):
        if self.action_idxes != []:
            bestmove = self.child_Q() + self.child_U()
            bestmove = self.action_idxes[
                np.argmax(bestmove[self.action_idxes])]
        else:
            bestmove = np.argmax(self.child_Q() + self.child_U())
        return bestmove

    def select_leaf(self):
        current = self
        while current.is_expanded:
            best_move = current.best_child()
            current = current.maybe_add_child(best_move)
        return current

    def add_dirichlet_noise(self, action_idxs, child_priors):
        valid_child_priors = child_priors[
            action_idxs]  # select only legal moves entries in child_priors array
        valid_child_priors = 0.75 * valid_child_priors + 0.25 * np.random.dirichlet(
            np.zeros([len(valid_child_priors)],
                     dtype=np.float32) + 192)
        child_priors[action_idxs] = valid_child_priors
        return child_priors

    def expand(self, child_priors):
        self.is_expanded = True
        action_idxs = self.game.actions()
        c_p = child_priors
        if not action_idxs:
            self.is_expanded = False
        self.action_idxes = action_idxs
        c_p[[i for i in range(len(child_priors)) if
             i not in action_idxs]] = 0.000000000  # mask all illegal actions
        if self.parent.parent is None:  # add dirichlet noise to child_priors in root node
            c_p = self.add_dirichlet_noise(action_idxs, c_p)
        self.child_priors = c_p

    def decode_n_move_pieces(self, board: Board, move):
        board.move(move)
        return board

    def maybe_add_child(self, move):
        if move not in self.children:
            copy_board = copy.deepcopy(self.game)  # make copy of board
            copy_board = self.decode_n_move_pieces(copy_board, move)
            self.children[move] = UCTNode(
                copy_board, move, parent=self)
        return self.children[move]

    def backup(self, value_estimate: float):
        current = self
        while current.parent is not None:
            current.number_visits += 1
            if current.game.player == 1:  # same as current.parent.game.player = 0
                current.total_value += (
                        1 * value_estimate)  # value estimate +1 = O wins
            elif current.game.player == 0:  # same as current.parent.game.player = 1
                current.total_value += (-1 * value_estimate)
            current = current.parent


class DummyNode(object):
    def __init__(self):
        self.parent = None
        self.child_total_value = collections.defaultdict(float)
        self.child_number_visits = collections.defaultdict(float)


def UCT_search(game_state, num_reads, net, temp):
    root = UCTNode(game_state, move=None, parent=DummyNode())
    for i in range(num_reads):
        leaf = root.select_leaf()
        encoded_s = encode_board(leaf.game)
        encoded_s = encoded_s.transpose(2, 0, 1)
        encoded_s = torch.from_numpy(encoded_s).float().cuda()
        child_priors, value_estimate = net(encoded_s)
        child_priors = child_priors.detach().cpu().numpy().reshape(-1);
        value_estimate = value_estimate.item()
        if leaf.game.check_winner() or leaf.game.actions() == []:  # if somebody won or draw
            leaf.backup(value_estimate)
            continue
        leaf.expand(child_priors)  # need to make sure valid moves
        leaf.backup(value_estimate)
    return root


def do_decode_n_move_pieces(board: Board, move):
    board.move(move)
    return board


def get_policy(root, temp=1):
    # policy = np.zeros([7], dtype=np.float32)
    # for idx in np.where(root.child_number_visits!=0)[0]:
    #    policy[idx] = ((root.child_number_visits[idx])**(1/temp))/sum(root.child_number_visits**(1/temp))
    return ((root.child_number_visits) ** (1 / temp)) / sum(
        root.child_number_visits ** (1 / temp))


def MCTS_self_play(connectnet, num_games, start_idx, cpu, args, iteration):
    if not os.path.isdir("./datasets/iter_%d" % iteration):
        if not os.path.isdir("datasets"):
            os.mkdir("datasets")
        os.mkdir("datasets/iter_%d" % iteration)

    for idxx in tqdm(range(start_idx, num_games + start_idx)):
        current_board = Board()
        checkmate = False
        dataset = []  # to get state, policy, value for neural network training
        states = []
        value = 0
        move_count = 0
        while checkmate == False and current_board.possible_actions() != []:
            if move_count < 11:
                t = args.temperature_MCTS
            else:
                t = 0.1
            states.append(copy.deepcopy(current_board.board))
            board_state = copy.deepcopy(encode_board(current_board))
            root = UCT_search(current_board, 777, connectnet, t)
            policy = get_policy(root, t)
            print("[CPU: %d]: Game %d POLICY:\n " % (cpu, idxx), policy)
            current_board = do_decode_n_move_pieces(current_board,
                                                    np.random.choice(np.array(
                                                        [0, 1, 2, 3, 4, 5, 6]), \
                                                        p=policy))  # decode move and move piece(s)
            dataset.append([board_state, policy])
            print("[Iteration: %d CPU: %d]: Game %d CURRENT BOARD:\n" % (
                iteration, cpu, idxx), current_board.board,
                  current_board.player)
            print(" ")
            if current_board.check_winner():  # if somebody won
                if current_board.player == 0:  # black wins
                    value = -1
                elif current_board.player == 1:  # white wins
                    value = 1
                checkmate = True
            move_count += 1
        dataset_p = []
        for idx, data in enumerate(dataset):
            s, p = data
            if idx == 0:
                dataset_p.append([s, p, 0])
            else:
                dataset_p.append([s, p, value])
        del dataset
        save_as_pickle("iter_%d/" % iteration + \
                       "dataset_iter%d_cpu%i_%i_%s" % (iteration, cpu, idxx,
                                                       datetime.datetime.today().strftime(
                                                           "%Y-%m-%d")),
                       dataset_p)


class BoardData(Dataset):
    def __init__(self, dataset):  # dataset = np.array of (s, p, v)
        self.X = dataset[:, 0]
        self.y_p, self.y_v = dataset[:, 1], dataset[:, 2]

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return np.int64(self.X[idx].transpose(2, 0, 1)), self.y_p[idx], \
               self.y_v[idx]



# b = Board()
# possible_actions = b.possible_actions()
# won = False
# while possible_actions and not won:
#     print(b.board)
#     r = b.move(column=random.randint(0, 6))
#     if r is "Invalid move":
#         continue
#     won = r
#     possible_actions = b.possible_actions()
# print(b.board)
