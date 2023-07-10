# -*- coding: utf-8 -*-
"""
Created on Thu May 18 18:11:21 2023

@author: dell

 Use deep reinforcement learning to learn Flappy Bird game
"""

# Import module
import random
import sys
import pygame
import torch
import torch.nn as nn
import matplotlib.pyplot as plt 
from IPython import display
from collections import deque
import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'
# This line because I received the error "OMP: Error #15: Initializing libiomp5md.dll, but found libiomp5md.dll already initialized."
# If you do not face this error you can delete this line

from classes.Bird import Bird
from classes.QTrainer import QTrainer



# All the Game Variables
WIDTH = 288
HEIGHT = 512
ELEVATION = HEIGHT * 0.8

framepersecond = 128
MAX_MEMORY = 100_000
BATCH_SIZE = 1000
GAMMA = 0.99
ITERATION = 1
LR = 0.001
MAX_EPSILON = 0.9      # Maximum epsilon value
MIN_EPSILON = 0.01 #0.001
EPSILON_DECAY   = 0.95


class FlappyBirdEnv:
    def __init__(self):
        self.window_width = WIDTH
        self.window_height = HEIGHT
        self.elevation = HEIGHT * 0.8
        self.pipeVelX = -6
        self.horizontal = int(WIDTH/5)
        self.vertical = int(HEIGHT/2)
        self.ground = 0
        self.mytempheight = 100
        self.population_last_round = []
        self.bird_list = []
        
        Bird.ELEVATION = ELEVATION
        
        # init display
        self.window = pygame.display.set_mode(
            (self.window_width, self.window_height))
        # Sets the title on top of game window
        pygame.display.set_caption('Flappy Bird Game')
        self.framepersecond_clock = pygame.time.Clock()
        pygame.init()
        self.load_assets()
        self.reset()

    def reset(self):
        # Generating two pipes for blitting on window
        first_pipe = self.createPipe()
        # List containing lower pipes
        self.down_pipes = [
            {'x': self.window_width + 150,
             'y': first_pipe[1]['y']}
        ]
        # List Containing upper pipes
        self.up_pipes = [
            {'x': self.window_width + 150,
             'y': first_pipe[0]['y']}
        ]

        self.your_score = 0
        self.bird = Bird(self.horizontal, self.vertical, self.birdHeight, self.birdWidth)
        self.bird_list = []

    def load_assets(self):
        pipeimage = 'images/pipe.png'
        background_image = 'images/background.png'
        birdplayer_image = 'images/bird.png'
        bird_upflap_image = 'images/bird-upflap.png'
        bird_downflap_image = 'images/bird-downflap.png'
        sealevel_image = 'images/base.png'
        self.game_images = {}
        # Load all the images which we will use in the game
        # images for displaying score
        self.game_images['scoreimages'] = (
            pygame.image.load('images/0.png').convert_alpha(),
            pygame.image.load('images/1.png').convert_alpha(),
            pygame.image.load('images/2.png').convert_alpha(),
            pygame.image.load('images/3.png').convert_alpha(),
            pygame.image.load('images/4.png').convert_alpha(),
            pygame.image.load('images/5.png').convert_alpha(),
            pygame.image.load('images/6.png').convert_alpha(),
            pygame.image.load('images/7.png').convert_alpha(),
            pygame.image.load('images/8.png').convert_alpha(),
            pygame.image.load('images/9.png').convert_alpha()
        )
        self.game_images['flappybird'] = pygame.image.load(
            birdplayer_image).convert_alpha()
        self.game_images['bird_upflap'] = pygame.image.load(
            bird_upflap_image).convert_alpha()
        self.game_images['bird_downflap'] = pygame.image.load(
            bird_downflap_image).convert_alpha()
        self.game_images['sea_level'] = pygame.image.load(
            sealevel_image).convert_alpha()
        self.game_images['background'] = pygame.image.load(
            background_image).convert_alpha()
        self.game_images['pipeimage'] = (pygame.transform.rotate(pygame.image.load(pipeimage)
                                                                 .convert_alpha(), 180), pygame.image.load(pipeimage).convert_alpha())
        self.pipeHeight = self.game_images['pipeimage'][0].get_height()
        self.pipeWidth = self.game_images['pipeimage'][0].get_width()
        self.birdHeight = self.game_images['flappybird'].get_height()
        self.birdWidth = self.game_images['flappybird'].get_width()
        self.seaHeight = self.game_images['sea_level'].get_height()

        
        self.sound = {
            'point_sound': pygame.mixer.Sound("audio/point.wav"),
            'hit_sound': pygame.mixer.Sound("audio/hit.wav"),
            'die_sound': pygame.mixer.Sound("audio/die.wav"),
            'swoosh_sound': pygame.mixer.Sound("audio/swoosh.wav"),
            'wing_sound': pygame.mixer.Sound("audio/wing.wav") }

    def createPipe(self):
        offset = self.window_height/2
        y2 = offset + \
            random.randrange(
                0, int(self.window_height - self.seaHeight - 1.2 * offset))
        pipeX = self.window_width + 10
        y1 = self.pipeHeight - y2 + offset
        pipe = [
            # upper Pipe
            {'x': pipeX, 'y': -y1},

            # lower Pipe
            {'x': pipeX, 'y': y2}
        ]
        return pipe
    
    def get_state(self, bird:Bird):
        # state x: array 4 elements
        # [velocity, x ditance to up_pipe, y distance to up_pipe, x distance to down_pipe, bird.x]
        x = []
        if(len(self.up_pipes) > 0 and len(self.down_pipes) > 0):
            x = [bird.velocity_y,
                 round((self.up_pipes[0]['x'] - bird.horizontal)/40),
                 round((bird.vertical - self.up_pipes[0]['y'] - self.pipeHeight)/40),
                 round((self.down_pipes[0]['y'] - bird.vertical - self.birdHeight)/40)]
            #x = [bird.velocity_y>0, self.up_pipes[0]['x']>bird.horizontal,self.up_pipes[0]['y'] > bird.vertical, self.down_pipes[0]['y'] > bird.vertical]
        return x
            

    def checkScore(self, bird: Bird):
        #when bird cross middle of the pipe, increment score by one
        playerMidPos = self.horizontal + self.birdWidth/2
        for pipe in self.up_pipes:
            pipeMidPos = pipe['x'] + self.pipeWidth/2
            if pipeMidPos <= playerMidPos < pipeMidPos + 4:
                bird.score += 1

    def checkGameOver(self, bird: Bird):
        horizontal = bird.horizontal
        vertical = bird.vertical

        if vertical > self.elevation- self.birdHeight or vertical < 0:
            bird.is_dead = True
            return True

        if len(self.up_pipes) > 0 and len(self.down_pipes) > 0:
            if((vertical < self.pipeHeight + self.up_pipes[0]['y'] or
                vertical + self.birdHeight > self.down_pipes[0]['y']) and
               (0 < self.up_pipes[0]['x'] - horizontal < self.birdWidth or
                    0    < horizontal - self.up_pipes[0]['x'] < self.pipeWidth)):
                bird.is_dead = True
                return True

        return False

    def displayScore(self, your_score):
        # Fetching the digits of score.
        numbers = [int(x) for x in list(str(your_score))]
        width = 0

        # finding the width of score images from numbers.
        for num in numbers:
            width += self.game_images['scoreimages'][num].get_width()
        Xoffset = (self.window_width - width)/1.1

        # Blitting the images on the window.
        for num in numbers:
            self.window.blit(self.game_images['scoreimages'][num],
                             (Xoffset, self.window_width*0.02))
            Xoffset += self.game_images['scoreimages'][num].get_width()

    def update_ui(self):
        # Lets blit our game images now
        self.window.blit(self.game_images['background'], (0, 0))
        for upperPipe, lowerPipe in zip(self.up_pipes, self.down_pipes):
            self.window.blit(self.game_images['pipeimage'][0],
                             (upperPipe['x'], upperPipe['y']))
            self.window.blit(self.game_images['pipeimage'][1],
                             (lowerPipe['x'], lowerPipe['y']))

        self.window.blit(
            self.game_images['sea_level'], (self.ground, ELEVATION))

        
        if (self.bird.velocity_y > 0):
            self.window.blit(
                self.game_images['bird_downflap'], (self.bird.horizontal, self.bird.vertical))
        else:
            self.window.blit(
                self.game_images['bird_upflap'], (self.bird.horizontal, self.bird.vertical))

        self.displayScore(self.your_score)
        # Refreshing the game window and displaying the score.
        pygame.display.update()
    
    def play_sound(self, sound_name:str):
        pygame.mixer.Sound.play(self.sound[sound_name+'_sound'])

    
        
        

    def run(self):
        print("WELCOME TO THE FLAPPY BIRD GAME")
        print("Press space to start the game")
        while True:
            for event in pygame.event.get():

                # if user clicks on cross button, close the game
                if event.type == pygame.QUIT or (event.type == pygame.KEYDOWN and
                                                 event.key == pygame.K_ESCAPE):
                    pygame.quit()
                    sys.exit()

                # If the user presses space or
                # up key, start the game
                elif event.type == pygame.KEYDOWN and (event.key == pygame.K_SPACE or
                                                       event.key == pygame.K_UP):
                    self.reset()
                    # start a new round
                    print('New round!')
                    while True:
                        self.play_step()

                # if user doesn't press anykey Nothing happen
                else:
                    self.update_ui()

    def play_step(self, action):
        #action: [0: nothing, 1: jump]
        
        #1.Collect the user input
        for event in pygame.event.get():
            if event.type == pygame.QUIT or (event.type == pygame.KEYDOWN and
                                             event.key == pygame.K_ESCAPE):
                pygame.quit()
                sys.exit()
                
        #2.Move
        self.bird.move(action)
        #3.Check if game over
        reward = 2 #if((bird.vertical < self.up_pipes[0]['y']+self.pipeHeight) and (self.down_pipes[0]['y'] < bird.vertical + self.birdHeight))
        game_over = False
        if(self.checkGameOver(self.bird)):
            reward = -10
            game_over = True
            self.play_sound('hit')
            return reward,game_over,self.your_score
        #4.Update score and pipes position
        for pipe in self.up_pipes:
            pipeMidPos = pipe['x'] + self.pipeWidth / 2
            if pipeMidPos <= self.horizontal < pipeMidPos + abs(self.pipeVelX):
                self.your_score += 1
                reward = 10
                self.play_sound('point')
                print(f"Your score is {self.your_score}")
                
        # move pipes to the left
        for upperPipe, lowerPipe in zip(self.up_pipes, self.down_pipes):
            upperPipe['x'] += self.pipeVelX
            lowerPipe['x'] += self.pipeVelX

        # Add a new pipe when the first is
        # about to cross the leftmost part of the screen
        if len(self.up_pipes) < 1 or (len(self.up_pipes) < 2 and 0 < self.up_pipes[0]['x'] < self.horizontal):
            newpipe = self.createPipe()
            self.up_pipes.append(newpipe[0])
            self.down_pipes.append(newpipe[1])

        # if the pipe is out of the screen, remove it
        if self.up_pipes[0]['x'] < -self.pipeWidth:
            self.up_pipes.pop(0)
            self.down_pipes.pop(0)

        #5.Update UI and clock                  
        self.update_ui()
        self.framepersecond_clock.tick(framepersecond)
        
        return reward,game_over,self.your_score




class Agent:
    def __init__(self):
        self.n_game = 0
        self.epsilon = MAX_EPSILON # Randomness
        self.gamma = GAMMA # discount rate
        self.memory = deque(maxlen=MAX_MEMORY) # popleft()
        self.model = nn.Sequential(
            nn.Linear(4,256),
            nn.Tanh(),
            nn.Linear(256,2)).cuda()
        self.trainer = QTrainer(self.model,lr=LR,gamma=self.gamma)


    def remember(self,state,action,reward,next_state,done):
        self.memory.append((state,action,reward,next_state,done)) # popleft if memory exceed

    def train_long_memory(self):
        if (len(self.memory) > BATCH_SIZE):
            mini_sample = random.sample(self.memory, BATCH_SIZE)
        else:
            mini_sample = self.memory
        states,actions,rewards,next_states,dones = zip(*mini_sample)
        self.trainer.train_step(states,actions,rewards,next_states,dones)

    def train_short_memory(self,state,action,reward,next_state,done):
        self.trainer.train_step(state,action,reward,next_state,done)

    def get_action(self,state, training=True):
        # actions: [0: do nothing, 1: jump]
        # random moves: tradeoff explotation / exploitation
        if(training and random.random()<self.epsilon):
            move = random.randint(0,1)  
        else:
            state0 = torch.tensor(state,dtype=torch.float).cuda()
            prediction = self.model(state0).cuda() # prediction by model 
            move = torch.argmax(prediction).item()
        # Decay epsilon
        if self.epsilon > MIN_EPSILON:
            self.epsilon *= EPSILON_DECAY
            self.epsilon = max(MIN_EPSILON, self.epsilon)
        return move
    
    def use_trained_model(self, path='model.pth'):
        self.model.load_state_dict(torch.load(path))


def plot(scores, mean_scores):
    display.clear_output(wait=True)
    display.display(plt.gcf())
    plt.clf()
    plt.title("Training...")
    plt.xlabel('Number of Games')
    plt.ylabel('Score')
    plt.plot(scores)
    plt.plot(mean_scores)
    plt.ylim(ymin=0)
    plt.text(len(scores)-1,scores[-1],str(scores[-1]))
    plt.text(len(mean_scores)-1,mean_scores[-1],str(mean_scores[-1]))
    plt.show(block=False)
    plt.pause(.1)
    
def train():
    plot_scores = []
    plot_mean_loss = []
    total_score = 0
    record = 0
    #agent = Agent()
    game = FlappyBirdEnv()
    n_steps = 0
    #while True:
    while agent.n_game <300:
        # Get Old state
        state_old = game.get_state(game.bird)

        # get move
        final_move = agent.get_action(state_old)

        # perform move and get new state
        reward, done, score = game.play_step(final_move)
        state_new = game.get_state(game.bird)
        n_steps+=1

        # train short memory
        agent.train_short_memory(state_old,final_move,reward,state_new,done)

        #remember
        agent.remember(state_old,final_move,reward,state_new,done)

        if done:
            n_steps = 0
            # Train long memory,plot result
            game.reset()
            agent.n_game += 1
            for i in range(ITERATION):
                agent.train_long_memory()
            if(score > record): # new High score 
                record = score
                torch.save(agent.model.state_dict(),'model.pth')
            print('Game:',agent.n_game,'Score:',score,'Record:',record)
            
            plot_scores.append(score)
            total_score+=score
            #mean_score = total_score / agent.n_game
            loss = agent.trainer.loss.item()
            plot_mean_loss.append(loss)
            plot(plot_scores,plot_mean_loss)

def play():
    #let the agent play this game, using the trained model
    agent = Agent()
    agent.use_trained_model()
    game = FlappyBirdEnv()
    
    plot_scores = []
    plot_mean_scores = []
    
    while True:
        # Get Old state
        state_old = game.get_state(game.bird)

        # get move
        final_move = agent.get_action(state_old, training=False)

        # perform move and get new state
        reward, done, score = game.play_step(final_move)

        if done:
            game.reset()
            agent.n_game += 1            
            plot_scores.append(score)
            mean_score = sum(plot_scores[-20:]) / len(plot_scores[-20:])
            plot_mean_scores.append(mean_score)
            if(agent.n_game%10==0):
                plot(plot_scores,plot_mean_scores)
            

# program where the game starts
if __name__ == "__main__":
    """
    First we let the model learn by itself in 300 games using train(), then see the result by play()
    """
    agent = Agent()
    train()
    
    #%%
    play()


    