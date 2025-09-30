# aigs

To get up and running:

1. Install `uv` so you can get our environment up and running (https://docs.astral.sh/uv/)
2. Fork this repo
3. Clone your fork
4. Sync `uv sync` in the repo.

Open the repo in your IDE. You can run the code with our dependencies using `uv run python main.py`.
Modify in `conf/config.yaml` to change the parameters you pass in `cfg`.
You can modify these through the cli: `uv run python main.py task=qd`, changes the value of task to `qd` from `mcts`.
For your own project, you can add our code as a dependency, with `uv add aigs` or `pip install aigs`.

## labs

- MCTS. Find a simple game (easier than chess or go) that can be played in the temrinal.
  1. Implement the game (do it in unity if you want) (connect four or checkers)
  2. Impement MCTS
  3. play with params and have a competation
- DRL (getting a good player)
  1. get unity ml-agent to run
  2. pick game. Use PPO. Finetune.
- quality diversity (finding good levels)
  1. implement map elite for levels
  2. create a dataset of different solvable levels
- llm lab

<!--## Exam-->

<!--1. Make your own game that is more awesome than default unity, and train ppo on it-->
