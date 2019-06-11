# EASY 21

This is the implementation of the game and the algorithms required in the assignments of 
the [RL course by David Silver](http://www0.cs.ucl.ac.uk/staff/d.silver/web/Teaching.html).

## Game Description
* The game is played with an infinite deck of cards. 
* Each draw from the deck results in a value between 1 and 10 with a 
colour of red or black
* At the start of the game both the player and the dealer draw one 
black card (fully observed) .
* Each turn the player may either stick or hit 
* If the player hits then she draws another card from the deck 
* If the player sticks she receives no further cards 
* The values of the player’s cards are added (black cards) or  \
subtracted (red cards) \
* If the player’s sum exceeds 21, or becomes less than 1, then she  \
“goes bust” and loses the game (reward -1) 
* If the player sticks then the dealer starts taking turns. If the  \
dealer goes bust, then the player wins; otherwise, the outcome – win  \
(reward +1), lose (reward -1), or draw (reward 0) – is the player  \
with the largest sum. 

The Dealer is a part of the Environment. You can try playing the game as a player, 
or try training the "Agent" to play the game using 3 different algorithms: 
MCMC Policy Iteration, Sarsa Lambda Policy Iteration and Linear Function Approximation.

Require Python 3.7.3 to run.

Run `python3 play_game.py` to play the game manually.
 
To try the RL algorithm, run `python3 play_game.py --ai` .

Run `python3 play_game.py --help` for more details.