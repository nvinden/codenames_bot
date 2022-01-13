import random


def game_finished(boards):
    game_finished_list = list()

    for board in boards:
        if not board['bomb'] or not board['blue'] or not board['red']:
            game_finished_list.append(True)
        else:
            game_finished_list.append(False)

    return game_finished_list

def blue_turn(input_board):
    board = input_board.copy()

    n_blue_left = len(board['blue'])
    n_red_left = len(board['red'])
    n_neutral_left = len(board['neutral'])
    n_bombs_left = len(board['bomb'])
    n_wrong_left = n_red_left + n_neutral_left + n_bombs_left

    n_going_for = random.randint(min(n_blue_left, 2), min(n_blue_left, 4))
    p_correct = [0.95, 0.75, 0.35, 0.05]
    fail_rates = [n_red_left / n_wrong_left, n_neutral_left / n_wrong_left + n_bombs_left / (n_wrong_left * 2), n_bombs_left / (n_wrong_left * 2)]

    n_blue = 0
    outcome = 'blue'

    for guess_number in range(n_going_for):
        percent = p_correct[guess_number]
        generated = random.uniform(0, 1)

        #wrong answer
        if generated > percent:
            fail_outcome = random.choices(["red", "neutral", "bomb"], weights = fail_rates)[0]
            board[fail_outcome].pop(random.randint(0, len(board[fail_outcome]) - 1))
            outcome = fail_outcome

            break
        #correct answer
        else:
            board['blue'].pop(random.randint(0, len(board['blue']) - 1))
            
        n_blue += 1

    return board

def game_result(input_board, last_turn : str):
    #returns: one in ["red_turn", "blue_turn", "red_win", "blue_win"]
    assert last_turn in ['red', 'blue']

    if not input_board['bomb'] and last_turn == "blue":
        return "red_win"
    if not input_board['bomb'] and last_turn == "red":
        return "blue_win"

    if not input_board['red']:
        return "red_win"
    if not input_board['blue']:
        return "blue_win"

    if last_turn == "blue":
        return "red_turn"
    if last_turn == "red":
        return "blue_turn"

def game_over(input_board):
    if not input_board['bomb']:
        return 1
    elif not input_board['red']:
        return 1
    elif not input_board['blue']:
        return 1
        
    return 0
