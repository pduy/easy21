from copy import deepcopy

import click

import easy21 as game
from easy21 import Dealer, draw, Color, Environment, State
from function_approx import LinearFunctionPlayer
from learning import Policy
from table_lookup import SarsaLambdaPlayer, MCPlayer

HELP = 'This is the card game Easy21 defined in the 2015 RL course by David ' \
       'Silver at UCL. \n\n' \
       '* The game is played with an infinite deck of cards \n' \
       '* Each draw from the deck results in a value between 1 and 10 with a ' \
       'colour of red or black \n' \
       'At the start of the game both the player and the dealer draw one ' \
       'black card (fully observed) \n' \
       '* Each turn the player may either stick or hit \n' \
       '* If the player hits then she draws another card from the deck \n' \
       '* If the player sticks she receives no further cards \n' \
       '* The values of the player’s cards are added (black cards) or ' \
       'subtracted (red cards)' \
       '* If the player’s sum exceeds 21, or becomes less than 1, then she ' \
       '“goes bust” and loses the game (reward -1) \n' \
       '* If the player sticks then the dealer starts taking turns. If the ' \
       'dealer goes bust, then the player wins; otherwise, the outcome – win ' \
       '(reward +1), lose (reward -1), or draw (reward 0) – is the player ' \
       'with the largest sum. \n'


@click.command()
@click.option('--ai/--human', default=False)
@click.option('--algo', default='sarsa', help='"sarsa", "mcmc" or "linear"')
@click.option('--slambda', default=0.5, help='lambda value in case of SARSA')
@click.option('--steps',
              default=10000,
              help='Number of training steps. Default 10000.')
def main(ai, algo, slambda, steps):
    if ai:
        ai_play(algo, slambda, steps)
    else:
        human_play()


def human_play():
    print(HELP)

    player, environment = game.init_basic_game()
    game_over = False

    print('GAME STARTS.')
    print(f'Dealer shows: {environment.dealer.val()}')
    print(f'You have: {player.val()}')

    while not game_over:
        print(f'\nNow choose an action: (0: "hit" or 1: "stick")')
        try:
            game_over = take_step_from_std_input(player, environment)
        except ValueError:
            print('Invalid action. Please choose again.')


def ai_play(algo, slambda, steps):
    player, environment = init_game(algo=algo, slambda=slambda)

    print(f'Starting with player {player}')
    print(f'Player {player} is training ...')

    player.train(steps)

    print('\n*******************FINISHED LEARNING ************************')
    print(' Now AI will show off ^_^ \n')

    for _ in range(10):
        demo_player, demo_environment = init_game(algo=algo)
        demo_player.policy = deepcopy(player.policy)
        demo(demo_player, demo_environment)


def demo(player, environment):
    print(f'\n* Initial state = {environment.state}\n')
    while not environment.state.is_terminal:
        action = player.choose_action()
        print(f'* {player} chooses {action}')

        environment.state = environment.step(player, action)
        print(f'* Current state = {environment.state}')

    report_final_result(player, environment.state)


def report_final_result(player, state: game.State):
    assert state.is_terminal
    if state.reward() == 1:
        print(f'MISSION ACCOMPLISHED: {player} WINS.')
    elif state.reward() == -1:
        print(f'GAME OVER: {player} GOES BUST.')
    else:
        print(f'GAME OVER: Draw.')


def take_step_from_std_input(player, environment):
    """
    1 action-step update in the cycle.

    Returns:
        Bool: Is game over.
    """
    action = int(input())
    if action == 1:
        print('Your turn is now over. Dealer turn ... ')
    next_state = environment.step(player, game.ACTIONS[action])
    print(f'Next state = {next_state}')

    if next_state.is_terminal:
        report_final_result(player, next_state)
        return True
    return False


def init_game(algo='sarsa', slambda=0.5):
    dealer = Dealer([draw(color=Color.black)])
    environment = Environment(state=State(), dealer=dealer)

    if algo == 'sarsa':
        player = SarsaLambdaPlayer(cards=[draw(color=Color.black)],
                                   policy=Policy(),
                                   environment=environment,
                                   sarsa_lambda=slambda)
    elif algo == 'mcmc':
        player = MCPlayer(cards=[draw(color=Color.black)],
                          policy=Policy(),
                          environment=environment)
    else:
        player = LinearFunctionPlayer(cards=[draw(color=Color.black)],
                                      policy=Policy(),
                                      environment=environment,
                                      sarsa_lambda=slambda)

    environment.state.player_score = player.val()
    environment.state.dealer_score = dealer.val()
    return player, environment


if __name__ == "__main__":
    main()
