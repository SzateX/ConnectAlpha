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
from torch import optim
from torch.nn.utils import clip_grad_norm_
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

import matplotlib.pyplot as plt

import torch.multiprocessing as mp

from argparse import ArgumentParser
import logging

USE_CUDA = True

def save_as_pickle(filename, data):
    complete_name = os.path.join("./datasets/",
                                 filename)
    with open(complete_name, 'wb') as output:
        pickle.dump(data, output)


def load_pickle(filename):
    complete_name = os.path.join("./datasets/",
                                 filename)
    with open(complete_name, 'rb') as pkl_file:
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
        self.player_won = 0

    def move(self, column: int):
        if self.board[0, column] != 0:
            return "Invalid move"
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
        self.player_won = self.player if won else 0
        self.player = 2 if self.player == 1 else 1
        return won

    def check_if_won(self, player: int, placed_row: int, placed_column: int):
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


class ConvolutionBlock(nn.Module):
    def __init__(self):
        super(ConvolutionBlock, self).__init__()
        self.action_size = 7
        self.convolution1 = nn.Conv2d(3, 128, 3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(128)

    def forward(self, s):
        s = s.view(-1, 3, 6, 7)  # batch_size x channels x board_x x board_y
        s = F.relu(self.bn1(self.convolution1(s)))
        return s


class ResBlock(nn.Module):
    def __init__(self, in_planes=128, planes=128, stride=1, downsample=None):
        super(ResBlock, self).__init__()
        self.convolution1 = nn.Conv2d(in_planes, planes, kernel_size=3,
                                      stride=stride,
                                      padding=1, bias=False)
        self.batch_normal1 = nn.BatchNorm2d(planes)
        self.convolution2 = nn.Conv2d(planes, planes, kernel_size=3,
                                      stride=stride,
                                      padding=1, bias=False)
        self.batch_normal2 = nn.BatchNorm2d(planes)

    def forward(self, x):
        residual = x
        out = self.convolution1(x)
        out = F.relu(self.batch_normal1(out))
        out = self.convolution2(out)
        out = self.batch_normal2(out)
        out += residual
        out = F.relu(out)
        return out


class OutBlock(nn.Module):
    def __init__(self):
        super(OutBlock, self).__init__()
        self.convolution = nn.Conv2d(128, 3, kernel_size=1)  # value head
        self.batch_normal = nn.BatchNorm2d(3)
        self.fc1 = nn.Linear(3 * 6 * 7, 32)
        self.fc2 = nn.Linear(32, 1)

        self.convolution1 = nn.Conv2d(128, 32, kernel_size=1)  # policy head
        self.batch_normal1 = nn.BatchNorm2d(32)
        self.log_soft_max = nn.LogSoftmax(dim=1)
        self.fc = nn.Linear(6 * 7 * 32, 7)

    def forward(self, s):
        v = F.relu(self.batch_normal(self.convolution(s)))  # value head
        v = v.view(-1, 3 * 6 * 7)  # batch_size X channel X height X width
        v = F.relu(self.fc1(v))
        v = torch.tanh(self.fc2(v))

        p = F.relu(self.batch_normal1(self.convolution1(s)))  # policy head
        p = p.view(-1, 6 * 7 * 32)
        p = self.fc(p)
        p = self.log_soft_max(p).exp()
        return p, v


class ConnectNet(nn.Module):
    def __init__(self):
        super(ConnectNet, self).__init__()
        self.convolution = ConvolutionBlock()
        for block in range(19):
            setattr(self, "res_%i" % block, ResBlock())
        self.out_block = OutBlock()

    def forward(self, s):
        s = self.convolution(s)
        for block in range(19):
            s = getattr(self, "res_%i" % block)(s)
        s = self.out_block(s)
        return s


# Neural Net loss function implemented via PyTorch
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
        action_idxs = self.game.possible_actions()
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

    def child_q(self):
        return self.child_total_value / (1 + self.child_number_visits)

    def child_u(self):
        return math.sqrt(self.number_visits) * (
                abs(self.child_priors) / (1 + self.child_number_visits))

    def best_child(self):
        if self.action_idxes:
            best_move = self.child_q() + self.child_u()
            # noinspection PyTypeChecker
            index: int = np.argmax(best_move[self.action_idxes])
            best_move = self.action_idxes[index]
        else:
            best_move = np.argmax(self.child_q() + self.child_u())
        return best_move

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
        action_indexes = self.game.possible_actions()
        c_p = child_priors
        if not action_indexes:
            self.is_expanded = False
        self.action_idxes = action_indexes
        c_p[[i for i in range(len(child_priors)) if
             i not in action_indexes]] = 0.000000000  # mask all illegal actions
        if self.parent.parent is None:  # add dirichlet noise to child_priors in root node
            c_p = self.add_dirichlet_noise(action_indexes, c_p)
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
        encoded_s = encoded_s.transpose((2, 0, 1))
        encoded_s = torch.from_numpy(encoded_s).float()
        if USE_CUDA:
            encoded_s = encoded_s.cuda()
        child_priors, value_estimate = net(encoded_s)
        child_priors = child_priors.detach().cpu().numpy().reshape(-1)
        value_estimate = value_estimate.item()
        if leaf.game.player_won or leaf.game.possible_actions == []:  # if somebody won or draw
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
    return (root.child_number_visits ** (1 / temp)) / sum(
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
                                                        [0, 1, 2, 3, 4, 5, 6]),
                                                        p=policy))  # decode move and move piece(s)
            dataset.append([board_state, policy])
            print("[Iteration: %d CPU: %d]: Game %d CURRENT BOARD:\n" % (
                iteration, cpu, idxx), current_board.board,
                  current_board.player)
            print(" ")
            if current_board.player_won:  # if somebody won
                if current_board.player == 1:  # black wins
                    value = -1
                elif current_board.player == 2:  # white wins
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


def run_MCTS(args, start_idx=0, iteration=0):
    net_to_play = "%s_iter%d.pth.tar" % (args.neural_net_name, iteration)
    net = ConnectNet()
    cuda = torch.cuda.is_available() if USE_CUDA else False
    if cuda:
        net.cuda()

    if args.MCTS_num_processes > 1:
        logger.info("Preparing model for multi-process MCTS...")
        mp.set_start_method("spawn", force=True)
        net.share_memory()
        net.eval()

        current_net_filename = os.path.join("./model_data/",
                                            net_to_play)
        if os.path.isfile(current_net_filename):
            checkpoint = torch.load(current_net_filename)
            net.load_state_dict(checkpoint['state_dict'])
            logger.info("Loaded %s model." % current_net_filename)
        else:
            torch.save({'state_dict': net.state_dict()},
                       os.path.join("./model_data/",
                                    net_to_play))
            logger.info("Initialized model.")

        processes = []
        if args.MCTS_num_processes > mp.cpu_count():
            num_processes = mp.cpu_count()
            logger.info(
                "Required number of processes exceed number of CPUs! Setting MCTS_num_processes to %d" % num_processes)
        else:
            num_processes = args.MCTS_num_processes

        logger.info("Spawning %d processes..." % num_processes)
        with torch.no_grad():
            for i in range(num_processes):
                p = mp.Process(target=MCTS_self_play, args=(
                    net, args.num_games_per_MCTS_process, start_idx, i, args,
                    iteration))
                p.start()
                processes.append(p)
            for p in processes:
                p.join()
        logger.info("Finished multi-process MCTS!")

    elif args.MCTS_num_processes == 1:
        logger.info("Preparing model for MCTS...")
        net.eval()

        current_net_filename = os.path.join("./model_data/",
                                            net_to_play)
        if os.path.isfile(current_net_filename):
            checkpoint = torch.load(current_net_filename)
            net.load_state_dict(checkpoint['state_dict'])
            logger.info("Loaded %s model." % current_net_filename)
        else:
            torch.save({'state_dict': net.state_dict()},
                       os.path.join("./model_data/",
                                    net_to_play))
            logger.info("Initialized model.")

        with torch.no_grad():
            MCTS_self_play(net, args.num_games_per_MCTS_process, start_idx, 0,
                           args, iteration)
        logger.info("Finished MCTS!")


class BoardData(Dataset):
    def __init__(self, dataset):  # dataset = np.array of (s, p, v)
        self.X = dataset[:, 0]
        self.y_p, self.y_v = dataset[:, 1], dataset[:, 2]

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return np.int64(self.X[idx].transpose(2, 0, 1)), self.y_p[idx], \
               self.y_v[idx]


def load_state(net, optimizer, scheduler, args, iteration,
               new_optim_state=True):
    """ Loads saved model and optimizer states if exists """
    base_path = "./model_data/"
    checkpoint_path = os.path.join(base_path, "%s_iter%d.pth.tar" % (
        args.neural_net_name, iteration))
    start_epoch, checkpoint = 0, None
    if os.path.isfile(checkpoint_path):
        checkpoint = torch.load(checkpoint_path)
    if checkpoint is not None:
        if (len(checkpoint) == 1) or (new_optim_state is True):
            net.load_state_dict(checkpoint['state_dict'])
            logger.info("Loaded checkpoint model %s." % checkpoint_path)
        else:
            start_epoch = checkpoint['epoch']
            net.load_state_dict(checkpoint['state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            scheduler.load_state_dict(checkpoint['scheduler'])
            logger.info(
                "Loaded checkpoint model %s, and optimizer, scheduler." % checkpoint_path)
    return start_epoch


def load_results(iteration):
    """ Loads saved results if exists """
    losses_path = "./model_data/losses_per_epoch_iter%d.pkl" % iteration
    if os.path.isfile(losses_path):
        losses_per_epoch = load_pickle(
            "losses_per_epoch_iter%d.pkl" % iteration)
        logger.info("Loaded results buffer")
    else:
        losses_per_epoch = []
    return losses_per_epoch


def train(net, dataset, optimizer, scheduler, start_epoch, cpu, args,
          iteration):
    torch.manual_seed(cpu)
    cuda = torch.cuda.is_available() if USE_CUDA else False
    net.train()
    criterion = AlphaLoss()

    train_set = BoardData(dataset)
    train_loader = DataLoader(train_set, batch_size=args.batch_size,
                              shuffle=True, num_workers=0, pin_memory=False)
    losses_per_epoch = load_results(iteration + 1)

    logger.info("Starting training process...")
    update_size = len(train_loader) // 10
    print("Update step size: %d" % update_size)
    for epoch in range(start_epoch, args.num_epochs):
        total_loss = 0.0
        losses_per_batch = []
        for i, data in enumerate(train_loader, 0):
            state, policy, value = data
            state, policy, value = state.float(), policy.float(), value.float()
            if cuda:
                state, policy, value = state.cuda(), policy.cuda(), value.cuda()
            policy_pred, value_pred = net(
                state)  # policy_pred = torch.Size([batch, 4672]) value_pred = torch.Size([batch, 1])
            loss = criterion(value_pred[:, 0], value, policy_pred, policy)
            loss = loss / args.gradient_acc_steps
            loss.backward()
            clip_grad_norm_(net.parameters(), args.max_norm)
            if (epoch % args.gradient_acc_steps) == 0:
                optimizer.step()
                optimizer.zero_grad()

            total_loss += loss.item()
            if i % update_size == (
                    update_size - 1):  # print every update_size-d mini-batches of size = batch_size
                losses_per_batch.append(
                    args.gradient_acc_steps * total_loss / update_size)
                print(
                    '[Iteration %d] Process ID: %d [Epoch: %d, %5d/ %d points] total loss per batch: %.3f' %
                    (iteration, os.getpid(), epoch + 1,
                     (i + 1) * args.batch_size, len(train_set),
                     losses_per_batch[-1]))
                print("Policy (actual, predicted):", policy[0].argmax().item(),
                      policy_pred[0].argmax().item())
                print("Policy data:", policy[0])
                print("Policy pred:", policy_pred[0])
                print("Value (actual, predicted):", value[0].item(),
                      value_pred[0, 0].item())
                # print("Conv grad: %.7f" % net.conv.conv1.weight.grad.mean().item())
                # print("Res18 grad %.7f:" % net.res_18.conv1.weight.grad.mean().item())
                print(" ")
                total_loss = 0.0

        scheduler.step()
        if len(losses_per_batch) >= 1:
            losses_per_epoch.append(
                sum(losses_per_batch) / len(losses_per_batch))
        if (epoch % 2) == 0:
            save_as_pickle("losses_per_epoch_iter%d.pkl" % (iteration + 1),
                           losses_per_epoch)
            torch.save({
                'epoch': epoch + 1,
                'state_dict': net.state_dict(),
                'optimizer': optimizer.state_dict(),
                'scheduler': scheduler.state_dict(),
            }, os.path.join("./model_data/",
                            "%s_iter%d.pth.tar" % (
                                args.neural_net_name, (iteration + 1))))
        '''
        # Early stopping
        if len(losses_per_epoch) > 50:
            if abs(sum(losses_per_epoch[-4:-1])/3-sum(losses_per_epoch[-16:-13])/3) <= 0.00017:
                break
        '''
    logger.info("Finished Training!")
    fig = plt.figure()
    ax = fig.add_subplot(222)
    ax.scatter(
        [e for e in range(start_epoch, (len(losses_per_epoch) + start_epoch))],
        losses_per_epoch)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss per batch")
    ax.set_title("Loss vs Epoch")
    plt.savefig(os.path.join("./model_data/", "Loss_vs_Epoch_iter%d_%s.png" % (
        (iteration + 1), datetime.datetime.today().strftime("%Y-%m-%d"))))
    plt.show()


def train_connectnet(args, iteration, new_optim_state):
    # gather data
    logger.info("Loading training data...")
    data_path = "./datasets/iter_%d/" % iteration
    datasets = []
    for idx, file in enumerate(os.listdir(data_path)):
        filename = os.path.join(data_path, file)
        with open(filename, 'rb') as fo:
            datasets.extend(pickle.load(fo, encoding='bytes'))
    datasets = np.array(datasets)
    logger.info("Loaded data from %s." % data_path)

    # train net
    net = ConnectNet()

    cuda = torch.cuda.is_available() if USE_CUDA else False
    if cuda:
        net.cuda()
    optimizer = optim.Adam(net.parameters(), lr=args.lr, betas=(0.8, 0.999))
    scheduler = optim.lr_scheduler.MultiStepLR(optimizer,
                                               milestones=[50, 100, 150, 200,
                                                           250, 300, 400],
                                               gamma=0.77)
    start_epoch = load_state(net, optimizer, scheduler, args, iteration,
                             new_optim_state)

    train(net, datasets, optimizer, scheduler, start_epoch, 0, args, iteration)


class Arena():
    def __init__(self, current_cnet, best_cnet):
        self.current = current_cnet
        self.best = best_cnet

    def play_round(self):
        logger.info("Starting game round...")
        if np.random.uniform(0, 1) <= 0.5:
            white = self.current
            black = self.best
            w = "current"
            b = "best"
        else:
            white = self.best
            black = self.current
            w = "best"
            b = "current"
        current_board = Board()
        checkmate = False
        dataset = []
        value = 0
        t = 0.1
        while checkmate == False and current_board.possible_actions() != []:
            dataset.append(copy.deepcopy(encode_board(current_board)))
            print("")
            print(current_board.board)
            if current_board.player == 1:
                root = UCT_search(current_board, 777, white, t)
                policy = get_policy(root, t)
                print("Policy: ", policy, "white = %s" % (str(w)))
            elif current_board.player == 2:
                root = UCT_search(current_board, 777, black, t)
                policy = get_policy(root, t)
                print("Policy: ", policy, "black = %s" % (str(b)))
            current_board = do_decode_n_move_pieces(current_board,
                                                    np.random.choice(np.array(
                                                        [0, 1, 2, 3, 4, 5, 6]),
                                                        p=policy))  # decode move and move piece(s)
            if current_board.player_won:  # someone wins
                if current_board.player == 1:  # black wins
                    value = -1
                elif current_board.player == 2:  # white wins
                    value = 1
                checkmate = True
        dataset.append(encode_board(current_board))
        if value == -1:
            dataset.append(f"{b} as black wins")
            return b, dataset
        elif value == 1:
            dataset.append(f"{w} as white wins")
            return w, dataset
        else:
            dataset.append("Nobody wins")
            return None, dataset

    def evaluate(self, num_games, cpu):
        current_wins = 0
        logger.info("[CPU %d]: Starting games..." % cpu)
        for i in range(num_games):
            with torch.no_grad():
                winner, dataset = self.play_round()
                print("%s wins!" % winner)
            if winner == "current":
                current_wins += 1
            save_as_pickle("evaluate_net_dataset_cpu%i_%i_%s_%s" % (
                cpu, i, datetime.datetime.today().strftime("%Y-%m-%d"),
                str(winner)), dataset)
        print("Current_net wins ratio: %.5f" % (current_wins / num_games))
        save_as_pickle("wins_cpu_%i" % (cpu),
                       {"best_win_ratio": current_wins / num_games,
                        "num_games": num_games})
        logger.info("[CPU %d]: Finished arena games!" % cpu)


def fork_process(arena_obj, num_games, cpu):  # make arena picklable
    arena_obj.evaluate(num_games, cpu)


def evaluate_nets(args, iteration_1, iteration_2):
    logger.info("Loading nets...")
    current_net = "%s_iter%d.pth.tar" % (args.neural_net_name, iteration_2)
    best_net = "%s_iter%d.pth.tar" % (args.neural_net_name, iteration_1)
    current_net_filename = os.path.join("./model_data/",
                                        current_net)
    best_net_filename = os.path.join("./model_data/",
                                     best_net)

    logger.info("Current net: %s" % current_net)
    logger.info("Previous (Best) net: %s" % best_net)

    current_cnet = ConnectNet()
    best_cnet = ConnectNet()
    cuda = torch.cuda.is_available() if USE_CUDA else False
    if cuda:
        current_cnet.cuda()
        best_cnet.cuda()

    if not os.path.isdir("./evaluator_data/"):
        os.mkdir("evaluator_data")

    if args.MCTS_num_processes > 1:
        mp.set_start_method("spawn", force=True)

        current_cnet.share_memory()
        best_cnet.share_memory()
        current_cnet.eval()
        best_cnet.eval()

        checkpoint = torch.load(current_net_filename)
        current_cnet.load_state_dict(checkpoint['state_dict'])
        checkpoint = torch.load(best_net_filename)
        best_cnet.load_state_dict(checkpoint['state_dict'])

        processes = []
        if args.MCTS_num_processes > mp.cpu_count():
            num_processes = mp.cpu_count()
            logger.info(
                "Required number of processes exceed number of CPUs! Setting MCTS_num_processes to %d" % num_processes)
        else:
            num_processes = args.MCTS_num_processes
        logger.info("Spawning %d processes..." % num_processes)
        with torch.no_grad():
            for i in range(num_processes):
                p = mp.Process(target=fork_process, args=(
                    Arena(current_cnet, best_cnet), args.num_evaluator_games,
                    i))
                p.start()
                processes.append(p)
            for p in processes:
                p.join()

        wins_ratio = 0.0
        for i in range(num_processes):
            stats = load_pickle("wins_cpu_%i" % (i))
            wins_ratio += stats['best_win_ratio']
        wins_ratio = wins_ratio / num_processes
        if wins_ratio >= 0.55:
            return iteration_2
        else:
            return iteration_1

    elif args.MCTS_num_processes == 1:
        current_cnet.eval()
        best_cnet.eval()
        checkpoint = torch.load(current_net_filename)
        current_cnet.load_state_dict(checkpoint['state_dict'])
        checkpoint = torch.load(best_net_filename)
        best_cnet.load_state_dict(checkpoint['state_dict'])
        arena1 = Arena(current_cnet=current_cnet, best_cnet=best_cnet)
        arena1.evaluate(num_games=args.num_evaluator_games, cpu=0)

        stats = load_pickle("wins_cpu_%i" % (0))
        if stats.best_win_ratio >= 0.55:
            return iteration_2
        else:
            return iteration_1


logging.basicConfig(format='%(asctime)s [%(levelname)s]: %(message)s',
                    datefmt='%m/%d/%Y %I:%M:%S %p', level=logging.INFO)
logger = logging.getLogger(__file__)

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--iteration", type=int, default=0,
                        help="Current iteration number to resume from")
    parser.add_argument("--total_iterations", type=int, default=1000,
                        help="Total number of iterations to run")
    parser.add_argument("--MCTS_num_processes", type=int, default=5,
                        help="Number of processes to run MCTS self-plays")
    parser.add_argument("--num_games_per_MCTS_process", type=int, default=120,
                        help="Number of games to simulate per MCTS self-play process")
    parser.add_argument("--temperature_MCTS", type=float, default=1.1,
                        help="Temperature for first 10 moves of each MCTS self-play")
    parser.add_argument("--num_evaluator_games", type=int, default=100,
                        help="No of games to play to evaluate neural nets")
    parser.add_argument("--neural_net_name", type=str,
                        default="cc4_current_net_", help="Name of neural net")
    parser.add_argument("--batch_size", type=int, default=32,
                        help="Training batch size")
    parser.add_argument("--num_epochs", type=int, default=300,
                        help="No of epochs")
    parser.add_argument("--lr", type=float, default=0.001,
                        help="learning rate")
    parser.add_argument("--gradient_acc_steps", type=int, default=1,
                        help="Number of steps of gradient accumulation")
    parser.add_argument("--max_norm", type=float, default=1.0,
                        help="Clipped gradient norm")
    args = parser.parse_args()

    logger.info("Starting iteration pipeline...")
    for i in range(args.iteration, args.total_iterations):
        run_MCTS(args, start_idx=0, iteration=i)
        train_connectnet(args, iteration=i, new_optim_state=True)
        if i >= 1:
            winner = evaluate_nets(args, i, i + 1)
            counts = 0
            while (winner != (i + 1)):
                logger.info(
                    "Trained net didn't perform better, generating more MCTS games for retraining...")
                run_MCTS(args, start_idx=(
                                                 counts + 1) * args.num_games_per_MCTS_process,
                         iteration=i)
                counts += 1
                train_connectnet(args, iteration=i, new_optim_state=True)
                winner = evaluate_nets(args, i, i + 1)
