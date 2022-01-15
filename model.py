import torch
import torch.nn as nn
import torch.nn.functional as F

import pandas as pd
import pickle

import os

from transformers import GPT2Tokenizer, GPT2ForSequenceClassification, GPT2Model, GPT2Config
from data import board2stringsender, board2stringreciever, create_uncategorized_board

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class SenderModel(torch.nn.Module):
    def __init__(self, n_sender_choices = 10000):
        super(SenderModel, self).__init__()

        self.max_length = 115
        self.n_sender_choices = n_sender_choices

        if not os.path.isdir("data/created"):
            os.mkdir("data/created")

        #Creating List of All Word Choices
        precompiled_path = os.path.join("data", "created", f"wiki_word_corpus_{n_sender_choices}.txt")
        if not os.path.isfile(precompiled_path):
            sender_choices_file_path = os.path.join("data", "base", "wiki_word_corpus.txt")
            df = pd.read_csv(sender_choices_file_path, sep = ' ', names = ["word", "occurances"])
            df = df.head(n_sender_choices)
            df = df.drop(columns = ["occurances", ])
            self.sender_choices = df['word'].tolist()

            with open(precompiled_path, "wb") as fp:
                pickle.dump(self.sender_choices, fp)
        else:
            with open(precompiled_path, "rb") as fp:
                self.sender_choices = pickle.load(fp)

        #Initializing GPT2 Elements
        self.configuration = GPT2Config.from_pretrained(os.path.join("data/base", "config.json"), output_hidden_states=False)
        #torch.save(self.configuration, os.path.join("data/base", "s_config.pt"))

        self.configuration.num_labels = n_sender_choices * 9

        self.tokenizer = GPT2Tokenizer.from_pretrained("data/base", bos_token='<|startoftext|>', eos_token='<|endoftext|>', pad_token='<|pad|>', add_special_tokens = False)
        #torch.save(self.tokenizer, os.path.join("data/base", "s_tok.pt"))
        self.tokenizer.add_special_tokens({"additional_special_tokens": ["<RED>", "<BLUE>", "<NEUTRAL>", "<BOMB>"]})

        self.gpt2 = GPT2ForSequenceClassification.from_pretrained(os.path.join("data/base", "pytorch_model.bin"), config=self.configuration)
        #torch.save(self.gpt2, os.path.join("data/base", "s_gpt2.pt"))
        self.gpt2.train()
        self.gpt2.resize_token_embeddings(len(self.tokenizer))
        self.gpt2.config.pad_token_id = self.tokenizer.pad_token_id

    def generate_discrete_answers(self, q_logits):
        out = list()

        for row in q_logits:
            max_index = torch.argmax(row).item()

            number = max_index // self.n_sender_choices + 1
            word_index = max_index % self.n_sender_choices
            word = self.sender_choices[word_index]

            out.append({"word": word, "num": number})

        return out

    def forward(self, board_list):
        if isinstance(board_list, dict):
            board_list = [board_list]

        board_str_list = [board2stringsender(board) for board in board_list]
        tokens_list = [self.tokenizer("<|startoftext|> " + board_str + "<|endoftext|>", truncation=True, max_length=self.max_length, padding="max_length") for board_str in board_str_list]

        b_input_ids = [torch.tensor(tokens['input_ids'], dtype = torch.long, device = device).unsqueeze(0) for tokens in tokens_list]
        b_masks = [torch.tensor(tokens['attention_mask'], dtype = torch.long, device = device).unsqueeze(0) for tokens in tokens_list]

        b_input_ids = torch.cat(b_input_ids, dim = 0)
        b_masks = torch.cat(b_masks, dim = 0)

        out = self.gpt2(b_input_ids, attention_mask = b_masks)
        q_logits = out['logits']

        discrete_answers = self.generate_discrete_answers(q_logits)

        return q_logits, discrete_answers

class RecieverModel(torch.nn.Module):
    def __init__(self, n_sender_choices = 25000):
        super(RecieverModel, self).__init__()

        self.max_length = 115
        self.n_sender_choices = n_sender_choices

        # initializing GPT2 elements
        self.configuration = GPT2Config.from_pretrained(os.path.join("data/base", "config.json"), output_hidden_states=False)
        #torch.save(self.configuration, os.path.join("data/base", "r_config.pt"))
        self.configuration.num_labels = 1

        self.tokenizer = GPT2Tokenizer.from_pretrained("data/base", bos_token='<|startoftext|>', eos_token='<|endoftext|>', pad_token='<|pad|>', add_special_tokens = False)
        #torch.save(self.tokenizer, os.path.join("data/base", "r_tok.pt"))
        self.tokenizer.add_special_tokens({"additional_special_tokens": ["<HINT>", "<NUMBER>", "<BOARD>", "<END>"]})

        self.gpt2 = GPT2Model.from_pretrained(os.path.join("data/base", "pytorch_model.bin"), config=self.configuration)
        #torch.save(self.gpt2, os.path.join("data/base", "r_gpt2.pt"))
        self.gpt2.train()
        self.gpt2.resize_token_embeddings(len(self.tokenizer))
        self.gpt2.config.pad_token_id = self.tokenizer.pad_token_id

        # creating linear units
        self.shared_representation_net = nn.Linear(self.max_length * self.configuration.hidden_size, 1)
        self.action_net_list = [nn.Linear(self.max_length * self.configuration.hidden_size, 1).to(device) for i in range(25)]

    def create_reciever_input_string(self, board_list, sender_input):
        uncat_board = create_uncategorized_board(board_list)
        out_str = board2stringreciever(uncat_board, sender_input['word'], sender_input['num'])
        return out_str

    def get_board_classifications(self, b_input_ids):
        mask = torch.zeros(b_input_ids.shape)

        for i, ids in enumerate(b_input_ids):
            for j, curr_id in enumerate(ids):
                if curr_id == self.board_token or curr_id == self.comma_token:
                    mask[i, j + 1] = 1

        return mask

    def forward(self, board_list, sender_input_list):
        if isinstance(board_list, dict):
            board_list = [board_list]

        uncat_board_list = [create_uncategorized_board(board) for board in board_list]
        n_tiles_list = [len(board) for board in uncat_board_list]
        board_str_list = [board2stringreciever(board, sender_input['word'], sender_input['num']) for board, sender_input in zip(uncat_board_list, sender_input_list)]
        tokens_list = [self.tokenizer("<|startoftext|> " + board_str + "<|endoftext|>", truncation=True, max_length=self.max_length, padding="max_length") for board_str in board_str_list]

        b_input_ids = [torch.tensor(tokens['input_ids'], dtype = torch.long, device = device).unsqueeze(0) for tokens in tokens_list]
        b_masks = [torch.tensor(tokens['attention_mask'], dtype = torch.long, device = device).unsqueeze(0) for tokens in tokens_list]

        b_input_ids = torch.cat(b_input_ids, dim = 0)
        b_masks = torch.cat(b_masks, dim = 0)

        out = self.gpt2(b_input_ids, attention_mask = b_masks)
        logits = out['last_hidden_state'].squeeze(-1).to(device)
        shared_representation = logits.view(logits.shape[0], -1)

        # shared state value
        #state_value = self.shared_representation_net(shared_representation).squeeze(1).to(device)

        # generating all action-dependednt q values
        q_values = torch.zeros([len(n_tiles_list), 25])
        for i, action_net in enumerate(self.action_net_list):
            curr_q = action_net(shared_representation).squeeze(1)
            q_values[:, i] = curr_q

        # creating list of selected actions via search
        action_list = list()
        for c_q, c_input, c_n_tiles, c_uncat_board in zip(q_values, sender_input_list, n_tiles_list, uncat_board_list):
            n_selections = c_input['num']
            selection_indexes = torch.topk(c_q[:c_n_tiles], min(n_selections, c_n_tiles)).indices.tolist()
            if not isinstance(selection_indexes, list):
                selection_indexes = [selection_indexes]
            action_list.append(selection_indexes)

        return q_values, action_list, uncat_board_list