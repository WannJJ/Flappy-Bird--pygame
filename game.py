# Import module
import random
import sys
import pygame
from pygame.locals import *
from pygame import QUIT, KEYDOWN, K_ESCAPE, K_SPACE, K_UP
import numpy as np


# Reference: https://www.geeksforgeeks.org/how-to-make-flappy-bird-game-in-pygame/

# All the Game Variables
#window_width = 600
#window_height = 499
window_width = 288
window_height = 512

# set height and width of window
window = pygame.display.set_mode((window_width, window_height))
elevation = window_height * 0.8
game_images = {}
framepersecond = 32
pipeimage = 'images/pipe.png'
background_image = 'images/background.png'
birdplayer_image = 'images/bird.png'
bird_upflap_image = 'images/bird-upflap.png'
bird_downflap_image = 'images/bird-downflap.png'
sealevel_image = 'images/base.png'
down_pipes = []
up_pipes = []

def flappygame():
    global up_pipes, down_pipes
    your_score = 0
    horizontal = int(window_width/5)
    vertical = int(window_height/2)
    ground = 0
    mytempheight = 100
    
    #Gernate new bird
    bird = Bird(horizontal, vertical)

	# Generating two pipes for blitting on window
    first_pipe = createPipe()
    second_pipe = createPipe()

	# List containing lower pipes

    down_pipes = [
        {'x': window_width,
        'y': first_pipe[1]['y']}
    ]

    # List Containing upper pipes
    up_pipes = [
        {'x': window_width,
        'y': first_pipe[0]['y']}
    ]

    # pipe velocity along x
    pipeVelX = -4



    while True:
        for event in pygame.event.get():
            if event.type == QUIT or (event.type == KEYDOWN and event.key == K_ESCAPE):
                pygame.quit()
                sys.exit()
            if event.type == KEYDOWN and (event.key == K_SPACE or event.key == K_UP):
                bird.flap()
                pygame.mixer.Sound.play(wing_sound)
                    
 
        # This function will return true
        # if the flappybird is crashed
        game_over = checkGameOver(bird)
        if game_over:
            pygame.mixer.Sound.play(hit_sound)
            print("GAME OVER!! Press space to restart the game")
            return

        # check for your_score
        for pipe in up_pipes:
            pipeMidPos = pipe['x'] + game_images['pipeimage'][0].get_width() / 2
            if pipeMidPos <= horizontal < pipeMidPos + abs(pipeVelX):
                your_score += 1
                pygame.mixer.Sound.play(point_sound)
                print(f"Your score is {your_score}")

        bird.update()

        # move pipes to the left
        for upperPipe, lowerPipe in zip(up_pipes, down_pipes):
            upperPipe['x'] += pipeVelX
            lowerPipe['x'] += pipeVelX

        # Add a new pipe when the first is
        # about to cross the leftmost part of the screen
        if len(up_pipes) < 1 or (len(up_pipes) <2 and 0 < up_pipes[0]['x'] < horizontal):
            newpipe = createPipe()
            up_pipes.append(newpipe[0])
            down_pipes.append(newpipe[1])

        # if the pipe is out of the screen, remove it
        if up_pipes[0]['x'] < -game_images['pipeimage'][0].get_width():
            up_pipes.pop(0)
            down_pipes.pop(0)

        # Lets blit our game images now
        window.blit(game_images['background'], (0, 0))
        for upperPipe, lowerPipe in zip(up_pipes, down_pipes):
            window.blit(game_images['pipeimage'][0],
                        (upperPipe['x'], upperPipe['y']))
                        
            window.blit(game_images['pipeimage'][1],
                        (lowerPipe['x'], lowerPipe['y']))
            

        window.blit(game_images['sea_level'], (ground, elevation))
        if (bird.velocity_y > 0):
            window.blit(game_images['bird_downflap'], (bird.horizontal, bird.vertical))
        else:
            window.blit(game_images['bird_upflap'], (bird.horizontal, bird.vertical))

        # Fetching the digits of score.
        numbers = [int(x) for x in list(str(your_score))]
        width = 0

        # finding the width of score images from numbers.
        for num in numbers:
            width += game_images['scoreimages'][num].get_width()
        Xoffset = (window_width - width)/1.1

        # Blitting the images on the window.
        for num in numbers:
            window.blit(game_images['scoreimages'][num],
                        (Xoffset, window_width*0.02))
            Xoffset += game_images['scoreimages'][num].get_width()

        # Refreshing the game window and displaying the score.
        pygame.display.update()
        framepersecond_clock.tick(framepersecond)


def isGameOver(horizontal, vertical, up_pipes, down_pipes):
    
	if vertical > elevation - 25 or vertical < 0:
		return True
	
	if len(up_pipes) > 0 and len(down_pipes) > 0:
		pipeHeight = game_images['pipeimage'][0].get_height()
		if((vertical < pipeHeight + up_pipes[0]['y'] or vertical + game_images['flappybird'].get_height() > down_pipes[0]['y']) and
        		(0 < up_pipes[0]['x'] - horizontal < game_images['flappybird'].get_width() or
		0 < horizontal - up_pipes[0]['x'] < game_images['pipeimage'][0].get_width())):
			return True

	
	return False

def createPipe():
	offset = window_height/3
	pipeHeight = game_images['pipeimage'][0].get_height()
	y2 = offset + \
		random.randrange(
			0, int(window_height - game_images['sea_level'].get_height() - 1.2 * offset))
	pipeX = window_width + 10
	y1 = pipeHeight - y2 + offset
	pipe = [
		# upper Pipe
		{'x': pipeX, 'y': -y1},

		# lower Pipe
		{'x': pipeX, 'y': y2}
	]
	return pipe

class Bird:

    def __init__(self, horizontal : int  , vertical : int) -> None:
        self.horizontal = horizontal
        self.vertical = vertical

        self.velocity_y = -9
        self.Max_Vel_Y = 10
        self.Min_Vel_Y = -8
        self.birdAccY = 1

        self.flap_velocity = -8
        self.flapped = False

        self.score = 0
        self.is_dead = False

        self.w1 = self.generate_wt(4, 6)
        self.b1 = self.generate_wt(1, 6)
        self.w2 = self.generate_wt(6, 1)
        self.b2 = self.generate_wt(1, 1)

        

    def is_dead(self) -> bool:
        return self.is_dead

    def update(self):
        global down_pipes, up_pipes
        
        if(len(up_pipes) > 0 and len(down_pipes) > 0):
            x = np.array([self.velocity_y, up_pipes[0]['x'] - self.horizontal, 
            up_pipes[0]['y'] - self.vertical, -down_pipes[0]['y'] + self.vertical])
            decision_score = self.forward(x, self.w1, self.b1, self.w2, self.b2)
            
            if(decision_score > 0):
                #self.flap()
                pass

        if self.velocity_y < self.Max_Vel_Y and not self.flapped:
            self.velocity_y += self.birdAccY

        if self.flapped:
            self.flapped = False
        playerHeight = game_images['flappybird'].get_height()
        self.vertical = self.vertical + min(self.velocity_y, elevation - self.vertical - playerHeight)

        
    def flap(self) -> None:
        if vertical > 0:
            self.velocity_y = self.flap_velocity
            self.flapped = True

    def checkScore(self, up_pipes):
        playerMidPos = self.horizontal + game_images['flappybird'].get_width()/2
        for pipe in up_pipes:
            pipeMidPos = pipe['x'] + game_images['pipeimage'][0].get_width()/2
            if pipeMidPos <= playerMidPos < pipeMidPos + 4:
                self.score += 1
    
    def sigmoid(x):
        return(1/(1 + np.exp(-x)))

    def generate_wt(self, x, y):
        l = []
        for i in range(x*y):
            l.append(np.random.randn())
        return (np.array(l).reshape(x, y))
    
    def forward(self, x, w1, b1, w2, b2):
        z1 = x.dot(w1) + b1
        #a1 = Bird.sigmoid(z1) 
        a1 = np.tanh(z1)

        
        # Output layer
        z2 = a1.dot(w2) + b2
        #a2 = Bird.sigmoid(z2)
        a2 = np.tanh(z2)
        
        return a2.sum()

                
def checkGameOver(bird: Bird):
    horizontal = bird.horizontal
    vertical = bird.vertical
    
    if vertical > elevation - 25 or vertical < 0:
        bird.is_dead = True
        return True
	
    if len(up_pipes) > 0 and len(down_pipes) > 0:
        pipeHeight = game_images['pipeimage'][0].get_height()
        if((vertical < pipeHeight + up_pipes[0]['y'] or vertical + game_images['flappybird'].get_height() > down_pipes[0]['y']) and
        (0 < up_pipes[0]['x'] - horizontal < game_images['flappybird'].get_width() or
        0 < horizontal - up_pipes[0]['x'] < game_images['pipeimage'][0].get_width())):
            bird.is_dead = True
            return True


	
        return False


# program where the game starts
if __name__ == "__main__":
    
	# For initializing modules of pygame library
    pygame.init()
    framepersecond_clock = pygame.time.Clock()

    # Sets the title on top of game window
    pygame.display.set_caption('Flappy Bird Game')

    # Load all the images which we will use in the game

    # images for displaying score
    game_images['scoreimages'] = (
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
    game_images['flappybird'] = pygame.image.load(birdplayer_image).convert_alpha()
    game_images['bird_upflap'] = pygame.image.load(bird_upflap_image).convert_alpha()
    game_images['bird_downflap'] = pygame.image.load(bird_downflap_image).convert_alpha()
    game_images['sea_level'] = pygame.image.load(
        sealevel_image).convert_alpha()
    game_images['background'] = pygame.image.load(
        background_image).convert_alpha()
    game_images['pipeimage'] = (pygame.transform.rotate(pygame.image.load(
        pipeimage).convert_alpha(), 180), pygame.image.load(
    pipeimage).convert_alpha())

    point_sound = pygame.mixer.Sound("audio/point.wav")
    hit_sound = pygame.mixer.Sound("audio/hit.wav")
    die_sound = pygame.mixer.Sound("audio/die.wav")
    smoosh_sound = pygame.mixer.Sound("audio/swoosh.wav")
    wing_sound = pygame.mixer.Sound("audio/wing.wav")


    print("WELCOME TO THE FLAPPY BIRD GAME")
    print("Press space to start the game")

    # sets the coordinates of flappy bird

    horizontal = int(window_width/5)
    vertical = int(
        (window_height - game_images['flappybird'].get_height())/2)
    ground = 0
    bird_list = []


    # Here starts the main game

    while True:


        while True:
            for event in pygame.event.get():

                # if user clicks on cross button, close the game
                if event.type == QUIT or (event.type == KEYDOWN and \
                                        event.key == K_ESCAPE):
                    pygame.quit()
                    sys.exit()

                # If the user presses space or
                # up key, start the game for them
                elif event.type == KEYDOWN and (event.key == K_SPACE or\
                                                event.key == K_UP):
                    flappygame()

                # if user doesn't press anykey Nothing happen
                else:                

                    window.blit(game_images['background'], (0, 0))
                    
                    window.blit(game_images['flappybird'],
                                (horizontal, vertical))
                    window.blit(game_images['sea_level'], (ground, elevation))                   
                    pygame.display.flip()
                    
                    pygame.display.update()
                    framepersecond_clock.tick(framepersecond)
