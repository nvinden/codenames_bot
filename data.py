import torch
import torch.nn as nn

from collections import namedtuple, deque

import random
import copy

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def epsilon_greedy_sender(q_sender, sender, epsilon = 0.9):
    p_values = torch.rand(size = (q_sender.shape[0], ), device = device)

    use_random_list = (p_values > epsilon)

    reciever_input_list = list()
    reciever_input_indexes_list = torch.empty([q_sender.shape[0], ], device = device)

    for batch_no, use_random in enumerate(use_random_list):
        if use_random:
            max_index = random.randint(0, sender.configuration.num_labels - 1)

            num = max_index // sender.n_sender_choices + 1
            word_index = max_index % sender.n_sender_choices
            word = sender.sender_choices[word_index]

            reciever_input_list.append({"word": word, "num": num})
            reciever_input_indexes_list[batch_no] = max_index
        else:
            max_index = torch.argmax(q_sender[batch_no]).item()

            num = max_index // sender.n_sender_choices + 1
            word_index = max_index % sender.n_sender_choices
            word = sender.sender_choices[word_index]

            reciever_input_list.append({"word": word, "num": num})
            reciever_input_indexes_list[batch_no] = max_index

    return reciever_input_list, reciever_input_indexes_list

def epsilon_greedy_reciever(q_reciever, sender_choices, board, epsilon = 0.9):
    p_values = torch.rand(size = (q_reciever.shape[0], ), device = device)
    use_random_list = (p_values > epsilon)

    reciever_output_list = list()

    for batch_no, use_random in enumerate(use_random_list):
        #random selection
        if use_random:
            number_of_selections = min(random.randrange(1, 9), len(board[batch_no]))
            curr_selections = random.sample(board[batch_no], number_of_selections)
            reciever_output_list.append(curr_selections)

        #true selection
        else:
            number_of_selections = min(sender_choices[batch_no]['num'], len(board[batch_no]))
            selection_indexes = torch.topk(q_reciever[batch_no, :len(board[batch_no])], number_of_selections)

            curr_selections = list()

            for index in selection_indexes[1]:
                index = index.item()

                curr_selections.append(board[batch_no][index])

            reciever_output_list.append(curr_selections)
        
    return reciever_output_list
      
def reward(boards, actions):
    new_boards = copy.deepcopy(boards)
    rewards = torch.zeros((len(boards)), device = device)

    for i, (board, action) in enumerate(zip(new_boards, actions)):
        curr_reward = 0.0
        for word in action:
            if word in board['red']:
                curr_reward += 0.2
                board['red'].remove(word)
            elif word in board['blue']:
                curr_reward -= 0.2
                board['blue'].remove(word)
                break
            elif word in board['neutral']:
                board['neutral'].remove(word)
                break
            elif word in board['bomb']:
                curr_reward -= 1.0
                board['bomb'].remove(word)
                break
            else:
                raise Exception("A word the reciever has chosen is not availble on the board")
        rewards[i] = curr_reward
    
    return new_boards, rewards
    
def board2stringsender(starting_board):
    str_out = "<RED> "
    for word in starting_board['red']:
        str_out += word + ", "
    str_out = str_out[0:-2]

    str_out += " <BLUE> "
    for word in starting_board['blue']:
        str_out += word + ", "
    str_out = str_out[0:-2]

    str_out += " <NEUTRAL> "
    for word in starting_board['neutral']:
        str_out += word + ", "
    str_out = str_out[0:-2]

    str_out += " <BOMB> "
    for word in starting_board['bomb']:
        str_out += word + ", "
    str_out = str_out[0:-2]
    
    str_out += " <DONE> "

    return str_out

def board2stringreciever(uncategorized_board_list, guess_word, guess_number):
    str_out = f"<HINT> {guess_word} <NUMBER> {guess_number} <BOARD> "

    for word in uncategorized_board_list:
        str_out += word + ", "
    str_out = str_out[0:-2]
    
    str_out += " <DONE> "

    return str_out

def create_uncategorized_board(board):
    uncat_board = list()

    uncat_board += board['red']
    uncat_board += board['blue']
    uncat_board += board['neutral']
    uncat_board += board['bomb']

    random.shuffle(uncat_board)

    return uncat_board

class StartingBoardGenerator():
    def __init__(self, vocab_list = "data/base/all_tiles.txt"):
        with open(vocab_list, "r") as f:
            self.tile_list = f.readlines()

        for i in range(len(self.tile_list)):
            self.tile_list[i] = self.tile_list[i].replace("\n", "")

        self.min_reds = 8
        self.min_blues = 8
        self.n_neutrals = 7
        self.n_bombs = 1

    def sample(self, starting_turn = 'red'):
        red_first = 1 if starting_turn == 'red' else 0

        total_pieces = self.min_reds + self.min_blues + self.n_neutrals + self.n_bombs + 1
        all_pieces = random.sample(self.tile_list, total_pieces)

        starting_board = dict()
        starting_board['red'] = [all_pieces.pop(0) for i in range(self.min_reds + red_first)]
        starting_board['blue'] = [all_pieces.pop(0) for i in range(self.min_blues + (1 - red_first))]
        starting_board['neutral'] = [all_pieces.pop(0) for i in range(self.n_neutrals)]
        starting_board['bomb'] = [all_pieces.pop(0) for i in range(self.n_bombs)]
        starting_board['first'] = red_first

        return starting_board

    def sample_batch(self, batch_size, starting_turn = 'red'):
        out = list()
        for batch in range(batch_size):
            out.append(self.sample(starting_turn = starting_turn))

        return out

Transition = namedtuple('Transition', ('state', 'sender_action', 'receiver_action', 'next_state', 'reward'))

class ReplayMemory(object):
    def __init__(self, capacity):
        self.memory = deque([],maxlen=capacity)

    def push(self, *args):
        """Save a transition"""
        self.memory.append(Transition(*args))

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)