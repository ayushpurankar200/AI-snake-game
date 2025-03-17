import random
import numpy as np
import pygame

from base import Base

class Snake(Base):
    def __init__(self, parent_screen, length = 5):
        
        super().__init__() # call super constructor from Base

        self.length = length # length is currently 5
        self.parent_screen = parent_screen
        self.block = pygame.image.load("resources/block.jpg")

        # initiate x and y coordinates
        self.x = [self.BLOCK_WIDTH] * self.length
        self.y = [self.BLOCK_WIDTH] * self.length

        # direction
        self.direction = "right"
    
    def draw(self):

        self.parent_screen.fill((0, 0, 0)) # clears screen to add all the elements

        for i in range(self.length):
            self.parent_screen.blit(self.block, (self.x[i], self.y[i])) # blit draws one object over another to redraw snake in new position

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

        self.draw()

# Apple instructions
class Apple(Base):
    def __init__(self, parent_screen):
        super().__init__()
        self.parent_screen = parent_screen
        self.apple_img = pygame.image.load("resources/apple.jpg")
        
        # first apple coordinates are (160, 200)
        self.x = self.BLOCK_WIDTH * 4
        self.y = self.BLOCK_WIDTH * 5

    def draw(self):
        self.parent_screen.blit(self.apple_img, (self.x, self.y))
    
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
        pygame.init() #initializes required modules in pygame
        pygame.display.set_caption("Snake Game - AI - Deep Q Learning")
        self.SCREEN_UPDATE = pygame.USEREVENT # pygame event is used to create a custom user event which can be used to trigger screen updates

        # change how fast the snake goes
        self.timer = 1
        pygame.time.set_timer(self.SCREEN_UPDATE, self.timer)

        self.surface = pygame.display.set_mode((self.SCREEN_SIZE, self.SCREEN_SIZE)) # sets the x and y vals for screen size

        self.snake = Snake(self.surface, length = 1)
        self.snake.draw()

        self.apple = Apple(parent_screen = self.surface)

        self.score = 0

        self.game_over = False

        self.reward = 0
    
    def play(self):

        pygame.time.set_timer(self.SCREEN_UPDATE, self.timer)

        self.snake.move()
        self.apple.draw()
        self.display_score()
        self.reward = -1 # negative award each move so that snake prioritizes shortest distance possible
    
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
    
    # Display scores
    def display_score(self):
        font = pygame.font.SysFont('arial', 20)
        msg = "Score: " + str(self.score)
        scores = font.render(f"{msg}", True, (200, 200, 200))
        self.surface.blit(scores, (480, 10))

    # Reset game
    def reset(self):
        self.snake = Snake(self.surface)
        self.apple = Apple(self.surface)
        self.score = 0
        self.game_over = False

    # Gets direction based on the 1 in a 4 len array
    def get_next_direction(self, move):
        # right down left up
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

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                quit()
            elif event.type == self.SCREEN_UPDATE:
                self.play()
                pygame.display.update()
                pygame.time.Clock().tick(200)
                break
        
        return self.reward, self.game_over, self.score