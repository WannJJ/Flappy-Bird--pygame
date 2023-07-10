# -*- coding: utf-8 -*-
"""
Created on Sat May 20 16:28:59 2023

@author: dell

 Create the model to play Flappy Bird using generic algorithm.
 After a few generations it can play trouble-free (can reach 100_000 points and still playing)
"""


# Import module
import random
import sys
import pygame
import numpy as np
from classes.Bird import Bird



# All the Game Variables
WIDTH = 288
HEIGHT = 512
ELEVATION = HEIGHT * 0.8

framepersecond = 64#32 
POPULATION = 1000



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
        self.generation = 0
        
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
        self.down_pipes = []
        self.up_pipes = []
        self.your_score = 0
        if (0 < len(self.bird_list) < POPULATION*0.8):
            new_list = []
            new_list.extend(self.bird_list)

            while len(new_list) < POPULATION*0.8:
                # 1. Parent selection
                father_id = random.randint(0, len(self.bird_list) - 1)
                mother_id = random.randint(0, len(self.bird_list) - 1)
                # 2. Crossover
                new_list.append(self.bird_list[father_id].cross(self.bird_list[mother_id]))
            self.bird_list = new_list

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

        self.point_sound = pygame.mixer.Sound("audio/point.wav")
        self.hit_sound = pygame.mixer.Sound("audio/hit.wav")
        self.die_sound = pygame.mixer.Sound("audio/die.wav")
        self.smoosh_sound = pygame.mixer.Sound("audio/swoosh.wav")
        self.wing_sound = pygame.mixer.Sound("audio/wing.wav")

    def createPipe(self):
        offset = self.window_height/3
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
        x = []
        if(len(self.up_pipes) > 0 and len(self.down_pipes) > 0):
            # state x: array 4 elements
            # [velocity, x ditance to up_pipe, y distance to up_pipe, x distance to down_pipe]
            x = [bird.velocity_y, self.up_pipes[0]['x'] - bird.horizontal,
                          self.up_pipes[0]['y'] - bird.vertical, -self.down_pipes[0]['y'] + bird.vertical]
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

        if vertical > self.elevation-25 or vertical < 0:
            bird.is_dead = True
            return True

        if len(self.up_pipes) > 0 and len(self.down_pipes) > 0:
            if((vertical < self.pipeHeight + self.up_pipes[0]['y'] or
                vertical + self.birdHeight > self.down_pipes[0]['y']) and
               (0 < self.up_pipes[0]['x'] - horizontal < self.birdWidth or
                    0 < horizontal - self.up_pipes[0]['x'] < self.pipeWidth)):
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

        for bird in self.bird_list:
            if (bird.velocity_y > 0):
                self.window.blit(
                    self.game_images['bird_downflap'], (bird.horizontal, bird.vertical))
            else:
                self.window.blit(
                    self.game_images['bird_upflap'], (bird.horizontal, bird.vertical))

        self.displayScore(self.your_score)
        # Refreshing the game window and displaying the score.
        pygame.display.update()
        self.framepersecond_clock.tick(framepersecond)

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
                    self.play()
                    self.generation += 1

                # if user doesn't press anykey Nothing happen
                else:
                    self.update_ui()

    def play(self):
        # start a new round
        print('New round!')
        # Gernate new birds
        while len(self.bird_list) < POPULATION:
            self.bird_list.append(
                Bird(self.horizontal, self.vertical, self.birdHeight, self.birdWidth))

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

        while True:
            for event in pygame.event.get():
                if event.type == pygame.QUIT or (event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE):
                    pygame.quit()
                    sys.exit()
                if event.type == pygame.KEYDOWN and (event.key == pygame.K_1 or event.key == pygame.K_RETURN):
                    return
                if event.type == pygame.KEYDOWN and (event.key == pygame.K_SPACE or event.key == pygame.K_UP):
                    self.bird_list[0].flap()

            for bird in (self. bird_list):
                bird.update(self.get_state(bird))
                self.checkGameOver(bird)
            self.bird_list = list(
                filter(lambda bird: not bird.is_dead, self.bird_list))

            # game over!!
            if(len(self.bird_list) <= max(5, POPULATION*0.05)):
                print(f'GENERATION {self.generation}:  score: {self.your_score}, population: {len(self.bird_list)}')
                return

            # check for your_score
            for pipe in self.up_pipes:
                pipeMidPos = pipe['x'] + self.pipeWidth / 2
                if pipeMidPos <= self.horizontal < pipeMidPos + abs(self.pipeVelX):
                    self.your_score += 1
                    pygame.mixer.Sound.play(self.point_sound)
                    #print(f"Your score is {self.your_score}")

                    if(self.your_score % 10 == 0):
                        if (len(self.population_last_round) >= self.your_score // 10):
                            print('Score:', self.your_score,"Population count:", len(self.bird_list),
                                  "(last time:", self.population_last_round[self.your_score//10 - 1], ")")
                            self.population_last_round[self.your_score //
                                                       10 - 1] = len(self.bird_list)
                        else:
                            print('Score:', self.your_score,"Population count:", len(self.bird_list))
                            self.population_last_round.append(
                                len(self.bird_list))
            bird.update(self.get_state(bird))

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

            self.update_ui()


# program where the game starts
if __name__ == "__main__":
    game = FlappyBirdEnv()
    game.run()
