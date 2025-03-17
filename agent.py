import json
import os
import random
import numpy as np
from game import Game
# from game_no_ui import Game
from collections import deque
from torch import nn, optim
import torch.nn.functional as F
import torch.types

# Artificial Neural Network
class ANN(nn.Module):
    def __init__(self, state_size, action_size):
        super(ANN, self).__init__()
        # fc1 takes input as state_size and outputs to 64 neurons in hidden layer.
        # output takes 64 neurons in hidden layer and transfers to out-put layer which has 4 nodes
        self.fc1 = nn.Linear(state_size, 64)                # state size is 16
        self.fc2 = nn.Linear(64, action_size)               # action size is 4

    # Passes the input data through the network
    def forward(self, state):
        x = self.fc1(state)                                 # passes to fc1
        x = F.relu(x)                                       # passes to relu function
        return self.fc2(x)                                  # passes through to output layer to find final value for each output

# Stores past rounds info
class ReplayMemory:
    def __init__(self, capacity):
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")    # uses GPU if available, otherwise CPU
        self.capacity = capacity                                                        # instantiates memory capacity
        self.memory = []

    # Adds the event - (state, action, reward, next_state, done) - to memory and manages memory space
    def push(self, event):
        self.memory.append(event)                           # appends event the memory
        if len(self.memory) > self.capacity:                # deletes oldest memory when capacity has been reached to make space for new one
            del self.memory[0]

    # Samples experiences by randomly selecting batch of k experiences
    def sample (self, k):
        experiences = random.sample(self.memory, k = k)     # creates random sample from memory
        
        # extract touple values from experiences. It stacks them vertically using vstack to then convert to pytorch tensor in float form. Finally, passes it to GPU/CPU
        states = torch.from_numpy(np.vstack([e[0] for e in experiences if e is not None])).float().to(self.device)
        actions = torch.from_numpy(np.vstack([e[1] for e in experiences if e is not None])).long().to(self.device)
        rewards = torch.from_numpy(np.vstack([e[2] for e in experiences if e is not None])).float().to(self.device)
        next_states = torch.from_numpy(np.vstack([e[3] for e in experiences if e is not None])).float().to(self.device)
        dones = torch.from_numpy(np.vstack([e[4] for e in experiences if e is not None]).astype(np.uint8)).float().to(self.device) # done is a boolean

        return states, actions, rewards, next_states, dones

# Hyperparameters
number_episodes = 100000                                     # number of episoddes the agent trains on
max_num_steps_per_ep = 200000                               # how long an episode lasts
epsilon_start_val = 1.0                                     # initial value for exploration/exploitation tradeoff, starts high to favor exploration
epsilon_end_val = 0.0001                                     # end value to favor exploitation
epsilon_decay_val = 0.99                                  # rate of epsilon decay
learning_rate = 0.01                                        # learning rate
minibatch_size = 100                                        # num of samples used for every training step
gamma = 0.95                                                # discount factor for future rewards
replay_buffer_size = int(1e5)                               # max capacity of replay memory
interpolation_parameter = 1e-2                              # parameter for soft updates in target network

state_size = 16                                             # input size
action_size = 4                                             # for each direction the snake can move
scores_on_100_episodes = deque(maxlen = 100)
folder = "model"

class Agent:
    def __init__(self, state_size, action_size):
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")            # gpu if available or cpu
        
        # set state and action sizes
        self.state_size = state_size
        self.action_size = action_size

        # set local and target networks
        self.local_network = ANN(state_size, action_size).to(self.device)
        self.target_network = ANN(state_size, action_size).to(self.device)
        self.optimizer = optim.Adam(self.local_network.parameters(), lr = learning_rate)
        self.memory = ReplayMemory(replay_buffer_size)
        self.t_step = 0
        self.record = -1
        self.epsilon = -1

    # Retrieves the state of the snake  (state = [Danger left, Danger up, Danger right-up, Danger right-down, Danger left-down, Danger left-up,
                                                # Direction left, Direction right, Direction up, Direction down, 
                                                # Food left, Food right, Food up, Food down])
    def get_state(self, game):
        # head coordinates
        head_x = game.snake.x[0]
        head_y = game.snake.y[0]

        # coordinates around snake head
        point_left = [(head_x - game.BLOCK_WIDTH), head_y]
        point_right = [(head_x + game.BLOCK_WIDTH), head_y]
        point_up = [head_x, (head_y - game.BLOCK_WIDTH)]
        point_down = [head_x, (head_y + game.BLOCK_WIDTH)]
        point_left_up = [(head_x - game.BLOCK_WIDTH), (head_y - game.BLOCK_WIDTH)]
        point_left_down = [(head_x - game.BLOCK_WIDTH), (head_y + game.BLOCK_WIDTH)]
        point_right_up = [(head_x + game.BLOCK_WIDTH), (head_y - game.BLOCK_WIDTH)]
        point_right_down = [(head_x + game.BLOCK_WIDTH), (head_y + game.BLOCK_WIDTH)]

        # instantiate state
        state = [
            # danger directions
            game.is_danger(point_left),
            game.is_danger(point_right),
            game.is_danger(point_up),
            game.is_danger(point_down),
            game.is_danger(point_left_up),
            game.is_danger(point_left_down),
            game.is_danger(point_right_up),
            game.is_danger(point_right_down),

            # move directions
            game.snake.direction == "left",
            game.snake.direction == "right",
            game.snake.direction == "up",
            game.snake.direction == "down",

            # apple location compared to head of snake
            game.apple.x < head_x,                          # left
            game.apple.x > head_x,                          # right
            game.apple.y > head_y,                          # up
            game.apple.y < head_y                           # down
        ]

        return np.array(state, dtype = int)

    def step(self, state, action, reward, next_state, done):
        self.memory.push((state, action, reward, next_state, done))                           # push's an event into the memory

        # agent will learn every 4 steps
        self.t_step = (self.t_step + 1) % 4
        if self.t_step == 0:
            if len(self.memory.memory) > minibatch_size:
                experiences = self.memory.sample(k = minibatch_size)                                # takes sample of 100 memories
                self.learn(experiences)                                                             # learn from experiences
    
    # Determines action taken at a step, will utilize the exploration-exploitation tradeoff
    def get_action(self, state, epsilon):
        state = torch.from_numpy(state).float().unsqueeze(0).to(self.device)                    # moves tensors to gpu/cpu as a batch
        self.local_network.eval()                                                               # puts Q network into evaluation mode to disable dropout and batch normalization which are used in training
        
        # ensures no gradients are calculated to save memory + computation
        with torch.no_grad():
            action_values = self.local_network(state)                                           # computes action values on local network

        self.local_network.train()                                                              # sets it back into train mode
        
        if random.random() > epsilon:
            move = torch.argmax(action_values).item()                                           # chooses best known action
        
        else: 
            move = random.randint(0, 3)

        return move

    # 
    def learn(self, experiences):
        states, actions, rewards, next_states, dones = experiences

        # bellman adaptation
        next_q_targets = self.target_network(next_states).detach().max(1)[0].unsqueeze(1)        # passes next state to the target network, detach makes sure it doesn't count gradients while max finds the maximum q value for each state accross all actions. unsqueeze reformats it for future uses
        q_targets = rewards + gamma * next_q_targets * (1 - dones)                               # no rewards added for after episode ends
        q_expected = self.local_network(states).gather(1, actions)                              # gathers q value for action taken in each step from predicted q values
        loss = F.mse_loss(q_expected, q_targets)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        self.soft_update(self.local_network, self.target_network)

    # 
    def soft_update(self, local_network, target_network):
        for local_params, target_params in zip(local_network.parameters(), target_network.parameters()):
            target_params.data.copy_(
                interpolation_parameter * local_params + (1.0 - interpolation_parameter) * target_params
            )

    # loads data from file specified in save_model
    def load(self, file_name='model.pth'):
        file_path = os.path.join(folder, file_name)
        if os.path.exists(file_path):
            self.local_network.load_state_dict(torch.load(file_path))
            print("Model Loaded")
            self.retrieve_data()
    
    # Saves current model into a file, creates a file if it doesn't already exist
    def save_model(self, file_name='model.pth'):
        if not os.path.exists(folder):
            os.mkdir(folder)

        file_name = os.path.join(folder, file_name)
        torch.save(self.local_network.state_dict(), file_name)                                  # saves agent progress to use later
    
    # Retrieves data from file and loads to record and epsilon
    def retrieve_data(self):
        file_name = "data.json"
        model_data_path = os.path.join(folder, file_name)
        if os.path.exists(model_data_path):
            with open(model_data_path, 'r') as file:
                data = json.load(file)

                if data is not None:
                    self.record = data['record']
                    self.epsilon = data['epsilon']
    
    # Saves metadata like current score, record, and epsilon to a json file
    def save_data(self, record, epsilon):
        file_name = "data.json"
        if not os.path.exists(folder):
            os.mkdir(folder)

        complete_path = os.path.join(folder, file_name)
        data = {'record': record, 'epsilon': epsilon}
        with open(complete_path, 'w') as file:
            json.dump(data, file, indent = 4)

if __name__ == "__main__":
    game = Game()
    agent = Agent(state_size = state_size, action_size = action_size)
    agent.load()
    max_score = 0

    epsilon = epsilon_start_val                                                                 # fixes epsilon if program stopped midway

    if agent.epsilon != -1:
        epsilon = agent.epsilon
        max_score = max(agent.record, max_score)

    print('epsilon starts at {}', epsilon)
    
    for episode in range(0, number_episodes):
        game.reset()
        score = 0

        for t in range(max_num_steps_per_ep):
            state_old = agent.get_state(game)
            action = agent.get_action(state_old, epsilon)

            move = [0, 0, 0, 0]
            move[action] = 1
            reward, done, score = game.run(move)  # Potentially stuck here

            # print(f"Step {t}: Action {action}, Reward {reward}, Done {done}")  # Debugging print

            state_new = agent.get_state(game)
            agent.step(state_old, action, reward, state_new, done)

            if done:
                print("Game Over. Next Episode:")
                break  # If done, restart the episode

        max_score = max(max_score, score)
        scores_on_100_episodes.append(score)
        epsilon = max(epsilon_end_val, epsilon_decay_val * epsilon)
        agent.save_model()
        agent.save_data(max_score, epsilon)

        print('Episode {}\t Curr Score {}\tMax Score {}\tAvg Score {:.2f}'.format(episode, score, max_score,
                                                                                      np.mean(scores_on_100_episodes)))
'''
        if episode % 50 == 0:
            print('Episode {}\t Curr Score {}\tMax Score {}\tAvg Score {:.2f}'.format(episode, score, max_score,
                                                                                      np.mean(scores_on_100_episodes)))
'''