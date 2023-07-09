# -*- coding: utf-8 -*-
"""
Created on Sun Jul  9 15:58:36 2023

@author: dell
"""
import numpy as np

class Bird:

    def __init__(self, horizontal:int, vertical:int, birdHeight:int, birdWidth:int) -> None:
        self.horizontal = horizontal
        self.vertical = vertical
        self.birdHeight = birdHeight
        self.birdWidth = birdWidth

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

    def update(self, state):
        # state x: np-array 4 elements
        # [velocity, x ditance to up_pipe, y distance to up_pipe, x distance to down_pipe]
        if (state):
            decision_score = self.forward(
                np.array(state), self.w1, self.b1, self.w2, self.b2)
            if(decision_score > 0):
                self.flap()


        if self.velocity_y < self.Max_Vel_Y and not self.flapped:
            self.velocity_y += self.birdAccY

        if self.flapped:
            self.flapped = False
        self.vertical = self.vertical + \
            min(self.velocity_y, self.ELEVATION - self.vertical - self.birdHeight)
    
    def move(self, flap:bool):
            if flap:
                self.flap()
            if self.velocity_y < self.Max_Vel_Y and not self.flapped:
                self.velocity_y += self.birdAccY
            if self.flapped:
                self.flapped = False
            self.vertical = self.vertical + \
                min(self.velocity_y, self.ELEVATION - self.vertical - self.birdHeight)

    def flap(self) -> None:
        if self.vertical > 0:
            self.velocity_y = self.flap_velocity
            self.flapped = True

    def sigmoid(x):
        return(1/(1 + np.exp(-x)))

    def generate_wt(self, x, y, epsilon=False):
        l = []
        for i in range(x*y):
            l.append(np.random.randn())
        l = np.array(l).reshape(x, y)
        if epsilon:
            l = l/100
        return l

    def forward(self, x, w1, b1, w2, b2):
        z1 = x.dot(w1) + b1
        #a1 = Bird.sigmoid(z1)
        a1 = np.tanh(z1)

        # Output layer
        z2 = a1.dot(w2) + b2
        #a2 = Bird.sigmoid(z2)
        a2 = z2.sum()
        return np.tanh(a2)

    def cross_wt(w1, w2):
        assert w1.shape == w2.shape
        mask = np.zeros(w1.shape[0]*w1.shape[1])
        mask[:round(len(mask) / 2)] = 1
        mask.reshape(w1.shape)
        np.random.shuffle(mask)

        mask = np.random.choice([0, 1], w1.shape)
        w_cross = w1 * mask + w2 * (np.ones(w1.shape) - mask)
        return w_cross

    def cross(self, x):
        # crossover
        child = Bird(self.horizontal, self.vertical,
                     self.birdHeight, self.birdWidth)
        child.b1 = Bird.cross_wt(self.b1, x.b1)
        child.b2 = Bird.cross_wt(self.b2, x.b2)
        child.w1 = Bird.cross_wt(self.w1, x.w1)
        child.w2 = Bird.cross_wt(self.w2, x.w2)

        return child
