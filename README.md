# MiniHex 

An [OpenAI gym](https://github.com/openai/gym/) environment that allows an 
agent to play the game of [Hex](https://en.wikipedia.org/wiki/Hex_(board_game)).
The aim for this environment is to be lean and have fast rollouts, as well as,
variable board size. With random actions it currently achieves *~250 games per 
second* in a 5x5 grid on a single core (Intel Xenon E3-1230 @3.3GHz).

Hex is a two player game and needs to be converted into a "single agent 
environment" to fit into the gym framework. We achieve this by requiring a
`opponent_policy` at creation time. Each move of the agent will be immediately
followed by a move of the opponent. This is a function that takes as input a
board state and outputs an action.

Following the implementation details of 
[Anthony et al.](https://arxiv.org/abs/1705.08439) we added a padded frame
of size 2 around the environment that is filled with black/white stones
respectively.

## Installation

~~pip install minihex~~ (TODO)

Editable installation (if you wish to tweak the environment):
```
git clone 
pip install -e minihex/
```

## Limitations

Currently the enviornment is missing the following features to go to version 1.0

- The swap action that is used to mitigate the disadvantage of playing second.
- RGB rendering mode
- add environment to pypi

## Bugs and Contributing
If you encounter problems, check the [GitHub issue page](https://github.com/FirefoxMetzger/minihex/issues) or open a new issue there.