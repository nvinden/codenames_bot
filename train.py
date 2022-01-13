from data import ReplayMemory, StartingBoardGenerator, epsilon_greedy_sender, epsilon_greedy_reciever, reward, Transition
from game import game_finished, blue_turn, game_over
from model import SenderModel, RecieverModel

import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter

import math
import sys

OPTIM_BATCH_SIZE = 8
batch_size = 6
starting_epsilon = 0.9
ending_epsilon = 0.05
n_episodes = 300
GAMMA = 0.8
TARGET_UPDATE = 10

# tensorboard logging
writer = SummaryWriter("Nick_Run1")

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def train():
    # creating the dataset
    board_manager = StartingBoardGenerator()

    # creating models
    policy_sender = SenderModel(n_sender_choices = 10000).to(device)
    policy_receiver = RecieverModel().to(device)

    target_sender = SenderModel().to(device)
    target_receiver = RecieverModel().to(device)
    target_sender.load_state_dict(policy_sender.state_dict())
    target_receiver.load_state_dict(policy_receiver.state_dict())

    # setting up optimizer
    optim_sender = torch.optim.RMSprop(policy_sender.parameters())
    optim_receiver = torch.optim.RMSprop(policy_receiver.parameters())
    
    epsilon = starting_epsilon
    epsilon_stepdown = (starting_epsilon - ending_epsilon) / n_episodes

    red_start = True

    # setting up memory
    memory = ReplayMemory(1000)

    for i_episode in range(n_episodes):
        print_detail_iteration = i_episode % 10 == 0 and i_episode != 0
        policy_target_switch = i_episode % TARGET_UPDATE == 0 and i_episode != 0

        hit_rate = math.sqrt(1 - epsilon)

        starting_colour = 'red' if red_start else 'blue'
        boards = board_manager.sample_batch(batch_size, starting_turn = starting_colour)
        red_turn = red_start

        turn_number = 1

        # game
        while boards:
            if red_turn: # red turn
                # one game turn
                q_sender, sender_choices = policy_sender(boards)
                sender_action, sender_action_indexes = epsilon_greedy_sender(q_sender, policy_sender, epsilon = hit_rate)
                q_reciever, action_list, uncategorized_board = policy_receiver(boards, sender_action)
                chosen = epsilon_greedy_reciever(q_reciever, sender_choices, uncategorized_board, epsilon = hit_rate)
                new_boards, rewards = reward(boards, chosen)

                if print_detail_iteration:
                    for rew in rewards:
                        writer.add_scalar("Reward", rew, i_episode)

                    board = boards[0]
                    rec_chose = chosen[0]
                    sen_chose = sender_choices[0]
                    rew = rewards[0]

                    print(f"turn {turn_number}")
                    print(f"    RED: {board['red']}")
                    print(f"    BLU: {board['blue']}")
                    print(f"    NEU: {board['neutral']}")
                    print(f"    BOM: {board['bomb']}")
                    print(f"    Sender word choice: {sen_chose['word']}")
                    print(f"    Sender word number: {sen_chose['num']}")
                    print(f"    Receiver choices: {rec_chose}")
                    print("    REWARD: {:.2f}".format(rew))

                for i in range(len(boards)):
                    memory.push(boards[i], sender_action_indexes[i], action_list[i], new_boards[i], rewards[i])

                boards = new_boards

                optimize_model_args = {
                    "memory": memory,
                    "optim_sender": optim_sender,
                    "optim_receiver": optim_receiver,
                    "policy_sender": policy_sender,
                    "policy_receiver": policy_receiver,
                    "target_sender": target_sender,
                    "target_receiver": target_receiver,
                    "epoch": i_episode
                }

                optimize_model(**optimize_model_args)
            else: # blue Turn
                boards = [blue_turn(board) for board in boards]

            # removing all of the already finished games
            games_done = game_finished(boards)
            boards = [boards[i] for i in range(len(boards)) if games_done[i] != True]

            red_turn = not red_turn
            turn_number += 1

        red_start = not red_start

        epsilon -= epsilon_stepdown

        if policy_target_switch:
            print("<POLICY-TARGET SWITCH>")
            target_sender.load_state_dict(policy_sender.state_dict())
            target_receiver.load_state_dict(policy_receiver.state_dict())
        
def optimize_model(**kwargs):
    memory = kwargs["memory"]
    optim_sender = kwargs["optim_sender"]
    optim_receiver = kwargs["optim_receiver"]
    policy_sender = kwargs["policy_sender"]
    policy_receiver = kwargs["policy_receiver"]
    target_sender = kwargs["target_sender"]
    target_receiver = kwargs["target_receiver"]
    epoch_number = kwargs["epoch"]

    if len(memory) < OPTIM_BATCH_SIZE:
        return

    # optimization for the sender
    transitions = memory.sample(OPTIM_BATCH_SIZE)
    batch = Transition(*zip(*transitions))

    sender_final_mask = torch.tensor([bool(game_over(game)) for game in batch.next_state], device = device)

    sender_action_batch = torch.stack(batch.sender_action).type(torch.int64)
    sender_reward_batch = torch.stack(batch.reward)

    sender_state_action_values, state_receiver_inputs = policy_sender(batch.state)
    sender_state_action_values = torch.stack([q[int(action.item())] for q, action in zip(sender_state_action_values, sender_action_batch)])

    q_logits, next_state_receiver_inputs = target_sender(batch.next_state)
    sender_next_state_values = torch.max(q_logits, dim = 1)[0]
    sender_next_state_values[sender_final_mask] = 0

    sender_expected_state_action_values = (sender_next_state_values * GAMMA) + sender_reward_batch

    criterion = nn.SmoothL1Loss()
    loss = criterion(sender_state_action_values, sender_expected_state_action_values)

    writer.add_scalar("Loss/Sender", loss.item(), epoch_number)

    sender_loss = loss.item()

    optim_sender.zero_grad()
    loss.backward()
    optim_sender.step()

    # optimization for the reciever
    receiver_final_mask = torch.tensor([bool(game_over(game)) for game in batch.next_state])
    receiver_reward_batch = torch.stack(batch.reward)

    receiver_state_action_values_final = torch.zeros(OPTIM_BATCH_SIZE, device = device)
    receiver_state_action_values = policy_receiver(batch.state, state_receiver_inputs)[0] #q (batch, 25, 1)
    for i, (action_list, q) in enumerate(zip(batch.receiver_action, receiver_state_action_values)):
        receiver_state_action_values_final[i] = q[action_list].mean()

    receiver_q_values = target_receiver(batch.next_state, next_state_receiver_inputs)[0]
    receiver_q_values_avg_top_num = torch.zeros(OPTIM_BATCH_SIZE, device = device)
    for i, q_batch in enumerate(receiver_q_values):
        num = next_state_receiver_inputs[i]['num']
        temp = torch.topk(q_batch, num).values
        receiver_q_values_avg_top_num[i] = temp.mean()
    receiver_q_values_avg_top_num[receiver_final_mask] = 0
    receiver_y = (receiver_q_values_avg_top_num * GAMMA) + receiver_reward_batch

    loss = criterion(receiver_y, receiver_state_action_values_final)

    writer.add_scalar("Loss/Receiver", loss.item(), epoch_number)

    rec_loss = loss.item()

    optim_receiver.zero_grad()
    loss.backward()
    optim_sender.step()

if __name__ == "__main__":
    train()