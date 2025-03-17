'''
This file is for the Snake AI. It contains no draw function as the snake only needs the numerical form of data. Visualization is for humans.
'''

import random

import numpy as np
from base import Base

class Snake(Base):
    def __init__(self, length = 5):
        
        super().__init__() # call super constructor from Base

        self.length = length # length is currently 1

        # initiate x and y coordinates
        self.x = [self.BLOCK_WIDTH] * self.length
        self.y = [self.BLOCK_WIDTH] * self.length

        # direction
        self.direction = "right"

    def increase(self):
        self.length += 1
        self.x.append(-1) # -1 is placeholder, gets updated when the snake moves
        self.y.append(-1)

# move the snake
    def move_left(self):
        if self.direction != 'right': # can't move the snake 180 degrees
            self.direction = 'left'

    def move_right(self):
        if self.direction != 'left':
            self.direction = 'right'

    def move_up(self):
        if self.direction != 'down':
            self.direction = 'up'

    def move_down(self):
        if self.direction != 'up':
            self.direction = 'down'

    # moves the snake to the position directly in front of it
    def move(self):
        
        for i in range(self.length - 1, 0, -1): # assigns position of each block to the space in front of it to show movement
            self.x[i] = self.x[i - 1]
            self.y[i] = self.y[i - 1]

        # top left corner is (0, 0), these if statements move the snake based on current direction
        if self.direction == 'right':
            self.x[0] += self.BLOCK_WIDTH

        if self.direction == 'left':
            self.x[0] -= self.BLOCK_WIDTH

        if self.direction == 'up':
            self.y[0] -= self.BLOCK_WIDTH

        if self.direction == 'down':
            self.y[0] += self.BLOCK_WIDTH

# Apple instructions
class Apple(Base):
    def __init__(self):
        super().__init__()
        
        self.x = self.BLOCK_WIDTH * 4
        self.y = self.BLOCK_WIDTH * 5

    def move(self, snake):
        # Creates and places food on map
        while True:
            x = random.randint(0, self.MAX_FOOD_INDEX) * self.BLOCK_WIDTH # randomly places it on the 20 possiblle locations
            y = random.randint(0, self.MAX_FOOD_INDEX) * self.BLOCK_WIDTH
            
            clean = True

            for i in range(0, snake.length): # checks new apple placement to see if it matches any of the snake's body segments
                if x == snake.x[i] and y == snake.y[i]:
                    clean = False
                    break
            
            if clean:
                self.x = x
                self.y = y
                return

# Base Game info
class Game(Base):
    def __init__(self):
        super().__init__()
        self.snake = Snake(length = 1)
        self.apple = Apple()
        self.score = 0
        self.game_over = False
        self.reward = 0
    
    def play(self):
        self.snake.move()
        self.reward = -0.1
    
        # check if snake eats apple
        if self.snake.x[0] == self.apple.x and self.snake.y[0] == self.apple.y:
            self.score += 1
            self.snake.increase()
            self.apple.move(self.snake)
            self.reward = 10

        if self.is_collision():
            self.game_over = True
            self.reward = -100

    def is_collision(self):
        # take the snake's head values
        head_x = self.snake.x[0]
        head_y = self.snake.y[0]

        # if snake hits itself return true
        for i in range(1, self.snake.length):
            if head_x == self.snake.x[i] and head_y == self.snake.y[i]:
                return True
        
        # if head hits screen limits return true
        if head_x > (self.SCREEN_SIZE - self.BLOCK_WIDTH) \
                or head_y > (self.SCREEN_SIZE - self.BLOCK_WIDTH) \
                or head_x < 0 \
                or head_y < 0:
            return True
        
        # return false as base
        return False

    def is_danger(self, point):
        # take the snake's head values
        point_x = point[0]
        point_y = point[1]

        # if snake hits itself return true
        for i in range(1, self.snake.length):
            if point_x == self.snake.x[i] and point_y == self.snake.y[i]:
                return True
        
        # if head hits screen limits return true
        if point_x > (self.SCREEN_SIZE - self.BLOCK_WIDTH) \
                or point_y > (self.SCREEN_SIZE - self.BLOCK_WIDTH) \
                or point_x < 0 \
                or point_y < 0:
            return True
        
        # return false as base
        return False

    # reset game
    def reset(self):
        self.snake = Snake()
        self.apple = Apple()
        self.score = 0
        self.game_over = False

    def get_next_direction(self, move):
        # right down left up
        new_dir = "right"
        if np.array_equal(move, [1, 0, 0, 0]):
            new_dir = "right"
        if np.array_equal(move, [0, 1, 0, 0]):
            new_dir = "down"
        if np.array_equal(move, [0, 0, 1, 0]):
            new_dir = "left"
        if np.array_equal(move, [0, 0, 0, 1]):
            new_dir = "up"
        
        return new_dir

    # run the game
    def run(self, move):
        dir = self.get_next_direction(move)

        if dir == "left":
            self.snake.move_left()
        elif dir == "right":
            self.snake.move_right()
        elif dir == "up":
            self.snake.move_up()
        elif dir == "down":
            self.snake.move_down()
        self.play()
        
        return self.reward, self.game_over, self.score