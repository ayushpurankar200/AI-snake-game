# AI-snake-game

This project is an AI-powered Snake game that utilizes Deep Q-Learning for training an agent to play the game efficiently. The implementation includes a graphical user interface (GUI) version and a non-GUI version for AI training purposes.

# Project Structure

game.py: Implements the graphical version of the Snake game using Pygame.

game_no_ui.py: A non-GUI version optimized for AI training, omitting visualization for performance.

agent.py: Implements the Deep Q-Learning agent using PyTorch, including a replay memory buffer and an artificial neural network.

base.py: Defines the base settings for the game, including screen size and block width.

# Dependencies

To run the project, install the following dependencies:

pip install pygame numpy torch

# Running the AI

To train the AI agent using Deep Q-Learning, run:

python agent.py

# AI Training Details

The agent uses a neural network with one hidden layer.

The replay memory mechanism helps the agent learn from past experiences.

Hyperparameters such as epsilon decay and learning rate can be modified in agent.py.

# Saving and Loading Models

The AI model is saved and loaded from the model/ directory. The agent stores its progress in model.pth and metadata like epsilon and highest score in data.json.

# Future Improvements

Enhance the neural network with more layers and optimizations.

Implement different reward mechanisms for more efficient learning.

Add reinforcement learning algorithms beyond Deep Q-Learning.
