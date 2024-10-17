# DDPG for Car Racing

This project implements a Deep Deterministic Policy Gradient (DDPG) algorithm to solve the CarRacing-v3 environment from OpenAI Gym. The agent learns to control a car around a racetrack, optimizing its driving performance over time.

## Project Structure

The project consists of a Jupyter notebook (`main.ipynb`) that contains the following key components:

1. Environment setup
2. Neural network architectures (Actor and Critic)
3. Replay Memory implementation
4. OUNoise for action exploration
5. DDPG Agent implementation
6. Training loop

## Requirements

To run this project, you need the following dependencies:

- Python 3.x
- PyTorch
- NumPy
- Gymnasium (OpenAI Gym)
- tqdm

You can install the required packages using pip:

```
pip install torch numpy gymnasium "gymnasium[atari, accept-rom-license]" gymnasium[box2d] tqdm
```

## Usage

1. Open the `main.ipynb` notebook in a Jupyter environment.
2. Run the cells in order to set up the environment, define the neural networks and agent, and start the training process.
3. The training progress will be displayed using tqdm, showing the current episode and average score.
4. After training, the model weights will be saved as checkpoint files.

## Key Components

### Actor and Critic Networks

The Actor and Critic networks are defined as PyTorch modules. They use convolutional layers to process the input state (an image of the race track) and fully connected layers to produce actions (Actor) or Q-values (Critic).

### Replay Memory

The ReplayMemory class implements experience replay, storing and sampling past experiences to improve learning stability.

### OUNoise

The OUNoise class implements the Ornstein-Uhlenbeck process for adding exploration noise to the agent's actions.

### DDPG Agent

The Agent class encapsulates the DDPG algorithm, including:
- Actor and Critic networks (and their target networks)
- Action selection
- Learning from experiences
- Soft updates of target networks

### Training Loop

The `train_agent` function implements the main training loop, which:
- Runs episodes of the CarRacing environment
- Collects experiences and trains the agent
- Tracks performance using a sliding window of scores
- Saves model checkpoints periodically and when solving the environment

## Hyperparameters

Key hyperparameters in the project include:
- Learning rates for Actor and Critic
- Replay buffer size
- Minibatch size
- Discount factor (gamma)
- Soft update parameter (tau)
- Noise parameters for exploration

These can be adjusted in the notebook to optimize performance.

## Performance

The agent's performance is measured by the average score over the last 100 episodes. The environment is considered solved when this average score reaches or exceeds 200.0.

## Saving and Loading Models

The training loop automatically saves model checkpoints:
- Every 100 episodes
- When the environment is solved
- At the end of training if not solved

You can load these checkpoints to continue training or to run the trained agent.

## Customization

Feel free to experiment with:
- Network architectures
- Hyperparameters
- Exploration strategies
- Reward shaping

## Troubleshooting

If you encounter CUDA-related issues, try setting the environment variable:
```python
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
```

This is already included in the notebook.

## Contributing

Contributions to improve the agent's performance or extend its capabilities are welcome. Please feel free to submit pull requests or open issues for discussion.
