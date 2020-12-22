# MiniHex 

An [OpenAI gym](https://github.com/openai/gym/) environment that allows an 
agent to play the game of [Hex](https://en.wikipedia.org/wiki/Hex_(board_game)).
The aim for this environment is to be lean and have fast rollouts, as well as,
variable board size. With random actions it currently achieves 
**~340 games per second** in a 11x11 grid (original size) on a single CPU 
(Intel Xenon E3-1230 @3.3GHz).

Hex is a two player game and needs to be converted into a "single agent 
environment" to fit into the gym framework. We achieve this by requiring a
`opponent_policy` at creation time. Each move of the agent will be immediately
followed by a move of the opponent. This is a function that takes as input a
board state and outputs an action.

## Installation

Get it from PyPi:

`pip install minihex`

Or an editable installation (if you wish to tweak the environment):
```
git clone https://github.com/FirefoxMetzger/minihex.git
pip install -e minihex/
```

## Minimal Working Example

```
import gym
import minihex


env = gym.make("hex-v0",
               opponent_policy=minihex.random_policy,
               board_size=11)

state, info = env.reset()
done = False
while not done:
    board, player = state
    action = minihex.random_policy(board, player, info)
    state, reward, done, info = env.step(action)

env.render()

if reward == -1:
    print("Player (Black) Lost")
elif reward == 1:
    print("Player (Black) Won")
else:
    print("Draw")

```

## Debug Mode

If the environment is instantiated with `debug=True` each step will check if a
valid action is provided, and an `IndexError` will be raised if an invalid one
is provided. This is very useful while writing agents, e.g., if the agent
maintains it's own belief over the environment and may request invalid actions.
When evaluating/running at scale, however, this check can cause significant
slowdown. Hence, it is only performed if explicitly requested.

## Limitations

Currently the enviornment is missing the following features to go to version 1.0

- The swap action that is used to mitigate the disadvantage of playing second.
- RGB rendering mode
- add environment to pypi
- no surrender action

## Bugs and Contributing
If you encounter problems, check the [GitHub issue page](https://github.com/FirefoxMetzger/minihex/issues) or open a new issue there.
