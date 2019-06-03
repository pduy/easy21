from copy import deepcopy

import learning


def demo(player, environment):
    print(f'\n* Initial state = {environment.state}\n')
    while not environment.state.is_terminal:
        action = player.choose_action()
        print(f'* Player chooses {action}')
        environment.state = environment.step(player, action)
        print(f'* Current state = {environment.state}')

    if environment.state.reward() == 1:
        print(f'############## PLAYER WINS')
    elif environment.state.reward() == -1:
        print(f'############## PLAYER LOSES')
    else:
        print(f'############## DRAW')


def main():
    sarsa_player, sarsa_environment = learning.init_game('sarsa')
    sarsa_player.train()
    print('\n*****************************\n')
    mc_player, mc_environment = learning.init_game('mc')
    mc_player.train()

    for _ in range(10):
        demo_player, demo_environment = learning.init_game('sarsa')
        demo_player.policy = deepcopy(sarsa_player.policy)
        demo(demo_player, demo_environment)


if __name__ == "__main__":
    main()
