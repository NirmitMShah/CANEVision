import pygame
import os
import math
import sys

import random
import neat
donkey[
    poop
import pickle

from neat import nn, population

screen_width = 1500
screen_height = 800
generation = 0
generation_threshold = 3
point_x = 1300
point_y = 400

#cane class 
class Cane:
  
    #attributes of cane object
    def __init__(self):
        self.surface = pygame.image.load("cane.png")
        self.surface = pygame.transform.scale(self.surface, (50, 50))
        self.rotate_surface = self.surface
        self.pos = [200, 400]
        #self.pos = [700, 50]
        self.angle = 0
        self.speed = 0
        self.center = [self.pos[0] + 50, self.pos[1] + 50]
        self.radars = []
        self.radars_for_draw = []
        self.is_alive = True
        self.goal = False
        self.distance = 0
        self.time_spent = 0
        self.hit_pink = False
        self.iteration_number = 3

    #drawing pygame
    def draw(self, screen):
        screen.blit(self.rotate_surface, self.pos)
        self.draw_radar(screen)

    #creating radars - effectively distances for lidar in real world 
    def draw_radar(self, screen):
        for r in self.radars:
            pos, dist = r
            pygame.draw.line(screen, (0, 255, 0), self.center, pos, 1)
            pygame.draw.circle(screen, (0, 255, 0), pos, 5)

    #checking if the cane collided to change if it is allive 
    def check_collision(self, map):
        self.is_alive = True
        for p in self.four_points:
            if map.get_at((int(p[0]), int(p[1]))) == (255, 255, 255):
                self.is_alive = False
                break

    #checking if objective is met 
    def check_final_collision(self, map):
        self.is_alive = True
        for p in self.four_points:
            if map.get_at((int(p[0]), int(p[1]))) == (255, 192, 203):
                self.hit_pink = True
                self.is_alive = False
                break

    #creating radars that measure distance to the nearest white from that set direction - once again these effectively simulate the lidar distance
    def check_radar(self, degree, map):
        len = 0
        x = int(self.center[0] + math.cos(math.radians(360 - (self.angle + degree))) * len)
        y = int(self.center[1] + math.sin(math.radians(360 - (self.angle + degree))) * len)

        while not map.get_at((x, y)) == (255, 255, 255, 255) and len < 450:
            len = len + 1
            x = int(self.center[0] + math.cos(math.radians(360 - (self.angle + degree))) * len)
            y = int(self.center[1] + math.sin(math.radians(360 - (self.angle + degree))) * len)

        dist = int(math.sqrt(math.pow(x - self.center[0], 2) + math.pow(y - self.center[1], 2)))
        self.radars.append([(x, y), dist])

    def update(self, map):

        #check speed
        self.speed = 15

        #check position
        self.rotate_surface = self.rot_center(self.surface, self.angle)
        self.pos[0] += math.cos(math.radians(360 - self.angle)) * self.speed
        if self.pos[0] < 20:
            self.pos[0] = 20
        elif self.pos[0] > screen_width - 120:
            self.pos[0] = screen_width - 120

        self.distance += self.speed
        self.time_spent += 1
        self.pos[1] += math.sin(math.radians(360 - self.angle)) * self.speed
        if self.pos[1] < 20:
            self.pos[1] = 20
        elif self.pos[1] > screen_height - 120:
            self.pos[1] = screen_height - 120

        # calculate 4 collision points
        self.center = [int(self.pos[0]) + 50, int(self.pos[1]) + 50]
        len = 40
        left_top = [self.center[0] + math.cos(math.radians(360 - (self.angle + 30))) * len, self.center[1] + math.sin(math.radians(360 - (self.angle + 30))) * len]
        right_top = [self.center[0] + math.cos(math.radians(360 - (self.angle + 150))) * len, self.center[1] + math.sin(math.radians(360 - (self.angle + 150))) * len]
        left_bottom = [self.center[0] + math.cos(math.radians(360 - (self.angle + 210))) * len, self.center[1] + math.sin(math.radians(360 - (self.angle + 210))) * len]
        right_bottom = [self.center[0] + math.cos(math.radians(360 - (self.angle + 330))) * len, self.center[1] + math.sin(math.radians(360 - (self.angle + 330))) * len]
        self.four_points = [left_top, right_top, left_bottom, right_bottom]
        self.check_green_collision(map)
        self.check_collision(map)
        self.radars.clear()
  
        for d in range(-90, 120, 45):
            self.check_radar(d, map)

    #passing in radar data into the Neural Network
    def get_data(self):
        radars = self.radars
        ret = [0, 0, 0, 0, 0]
        #ret = [0, 0 ,0, 0, 0]
        for i, r in enumerate(radars):
            ret[i] = int(r[1] / 30)
        return ret

    def hitting_pink(self):
        return self.hit_pink

    def timer(self):
        return self.time_spent

    def get_alive(self):
        return self.is_alive

    #Reward function rewarding for smaller distance and punishing for spending time 
    def get_reward(self):
        dist = int(math.sqrt(math.pow(point_x - self.center[0], 2) + math.pow(point_y - self.center[1], 2)))

        return 50 / dist - self.time_spent / 2000

    def rot_center(self, image, angle):
        orig_rect = image.get_rect()
        rot_image = pygame.transform.rotate(image, angle)
        rot_rect = orig_rect.copy()
        rot_rect.center = rot_image.get_rect().center
        rot_image = rot_image.subsurface(rot_rect).copy()
        return rot_image

    #def turning_punishment(self):

#This is the function to test running an already trained model
def pickle_cane(config):
    cane = Cane()
    with open('Pickle_F_Map.py', 'rb') as file:
        loaded_model = pickle.load(file)
    input_data = cane.get_data()
    #print(input_data)
    neat_net = neat.nn.FeedForwardNetwork.create(loaded_model, config)

    pygame.init()
    screen = pygame.display.set_mode((screen_width, screen_height))
    clock = pygame.time.Clock()
    generation_font = pygame.font.SysFont("Arial", 70)
    font = pygame.font.SysFont("Arial", 30)
    map = pygame.image.load('f_map5.png')

    while True:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                sys.exit(0)
        print(cane.get_data())
        output = neat_net.activate(cane.get_data())
        i = output.index(max(output))
        if i == 0:
            cane.angle += 10
        else:
            cane.angle -= 10
        cane.update(map)
        remain_canes = 0
        if cane.get_alive():
            remain_canes += 1
            cane.update(map)
        if remain_canes == 0:
            break

        screen.blit(map, (0, 0))

        if cane.get_alive():
            cane.draw(screen)

        text = generation_font.render("Generation : " + str(generation), True, (255, 255, 0))
        text_rect = text.get_rect()
        text_rect.center = (screen_width / 2, 100)
        screen.blit(text, text_rect)

        text = font.render("remain canes : " + str(remain_canes), True, (0, 0, 0))
        text_rect = text.get_rect()
        text_rect.center = (screen_width / 2, 200)
        screen.blit(text, text_rect)

        pygame.display.flip()
        clock.tick(0)


#this is the function that actually trained the model
def run_cane(genomes, config):
    #init NEAT
    nets = []
    canes = []

    #creating Neural Networks and giving each cane one
    for id, g in genomes:
        net = neat.nn.FeedForwardNetwork.create(g, config)
        nets.append(net)
        g.fitness = 0

        #init my canes
        canes.append(Cane())

    #init the digital environment
    pygame.init()
    screen = pygame.display.set_mode((screen_width, screen_height))
    clock = pygame.time.Clock()
    #generation_font = pygame.font.SysFont("Arial", 70)
    font = pygame.font.SysFont("Arial", 30)
    map = pygame.image.load('mapTest.png')
    breaking = False


    #main loop that dictates all the canes in a generation
    global generation
    global iteration
    global line
    #generation += 1
    while True:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                sys.exit(0)

        #input data and get result from network
        for index, cane in enumerate(canes):
            #if cane.timer() % 100 == 0:
            output = nets[index].activate(cane.get_data())
            i = output.index(max(output))
                #print(cane.get_data())

            if i == 0:
                cane.angle += 10
                #genomes[i][1].fitness -= ()
            if i == 1:
                cane.angle -= 10
            if i == 2:
                cane.angle += 0
                #genomes[i][1].fitness += 2*cane.get_reward()

        #update cane and fitness
        remain_canes = 0
        for i, cane in enumerate(canes):
            if cane.get_alive():
                remain_canes += 1
                cane.update(map)
                genomes[i][1].fitness += cane.get_reward()
                #print(cane.get_reward())
                #Fitness = genomes[i][1].fitness
                #print(Fitness)

      #break out of the loop when the cane reaches the end or spends too much time
            if cane.hitting_pink():
                genomes[i][1].fitness += 100
                cane.update(map)
                generation += 1
                breaking = True
            if cane.timer() >= 900:
                breaking = True
                generation = 0


        if breaking:
            break

        if remain_canes == 0:
            generation = 0
            break


        #drawing
        screen.blit(map, (0, 0))
        for cane in canes:
            if cane.get_alive():
                cane.draw(screen)

        pygame.display.flip()
        clock.tick(0)


def main():
    
    #setting up config path
    config_path = "config-feedforward.txt"
    config = neat.config.Config(neat.DefaultGenome, neat.DefaultReproduction,
                                neat.DefaultSpeciesSet, neat.DefaultStagnation, config_path)
    config.save_best = True
    #config.checkpoint_time_interval = 3

    #Neat Statistics to see how the generation did
    pop = population.Population(config)
    stats = neat.StatisticsReporter()
    pop.add_reporter(stats)
    pop.add_reporter(neat.StdOutReporter(True))
    pop.add_reporter(neat.StatisticsReporter())
    pop.add_reporter(neat.Checkpointer(2))

    #winner is the converged trained model
    winner = pop.run(run_cane, 75)
  
    #pop.run(pickle_cane(config)) - this was testing the if the trained model worked
  
    #pickle the trained model to allow for access on the raspberry pi
    with open('timeLapse', 'wb') as f:
        pickle.dump(winner, f)
        f.close()

    #predictions = neat_net.activate(input_data)
    #print(predictions)


if __name__ == "__main__":
    main()

