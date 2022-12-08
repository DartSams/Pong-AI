import pygame
from pong import Game
import os
import neat
import pickle

class PongGame:
    def __init__(self,window,width,height):
        self.game = Game(window,width,height)
        self.left_paddle = self.game.left_paddle
        self.right_paddle = self.game.right_paddle
        self.ball = self.game.ball


    def test_ai(self,genome,config):
        net = neat.nn.FeedForwardNetwork.create(genome,config)

        RUN = True
        CLOCK = pygame.time.Clock()
        FPS = 60
        while RUN:
            CLOCK.tick(FPS)
            for event in pygame.event.get(): #all pygame events
                if event.type == pygame.QUIT: #if red X is clicked ends game
                    RUN = False
                    break

            keys = pygame.key.get_pressed() #gets all keyboard presses
            if keys[pygame.K_w]:
                self.game.move_paddle(left=True,up=True)

            if keys[pygame.K_s]:
                self.game.move_paddle(left=True,up=False)

            ai_output = net.activate((self.right_paddle.y,self.ball.y,abs(self.right_paddle.x-self.ball.x)))#takes the inputs as parameters sends to hidden layers and then outputs a number for each output (0,0,456) 1st = stand still,2nd = move up,3rd = move down
            # print(output1,output2)
            decision2 = ai_output.index(max(ai_output)) #return the index of the max number in a list

            if decision2 == 0:
                pass
            if decision2 == 1:
                self.game.move_paddle(left=False,up=True)
            if decision2 == 2:
                self.game.move_paddle(left=False,up=False)

            self.game.loop()
            self.game.draw()
            pygame.display.update()

        pygame.quit()

    def train_ai(self,genome1,genome2,config):
        net1 = neat.nn.FeedForwardNetwork.create(genome1,config) #creates a neural network for the genome
        net2 = neat.nn.FeedForwardNetwork.create(genome2,config)
        
        RUN = True
        CLOCK = pygame.time.Clock()
        FPS = 60
        while RUN:
            # CLOCK.tick(FPS)
            for event in pygame.event.get(): #all pygame events
                if event.type == pygame.QUIT: #if red X is clicked ends game
                    quit()

            output1 = net1.activate((self.left_paddle.y,self.ball.y,abs(self.left_paddle.x-self.ball.x)))#takes the inputs as parameters sends to hidden layers and then outputs a number for each output (0,0,456) 1st = stand still,2nd = move up,3rd = move down
            output2 = net2.activate((self.right_paddle.y,self.ball.y,abs(self.right_paddle.x-self.ball.x)))#takes the inputs as parameters sends to hidden layers and then outputs a number for each output (0,0,456) 1st = stand still,2nd = move up,3rd = move down
            # print(output1,output2)
            decision1 = output1.index(max(output1)) #return the index of the max number in a list
            decision2 = output2.index(max(output2)) #return the index of the max number in a list

            if decision1 == 0:
                pass
            if decision1 == 1:
                self.game.move_paddle(left=True,up=True)
            if decision1 == 2:
                self.game.move_paddle(left=True,up=False)

            if decision2 == 0:
                pass
            if decision2 == 1:
                self.game.move_paddle(left=False,up=True)
            if decision2 == 2:
                self.game.move_paddle(left=False,up=False)
            

            game_info = self.game.loop()
            self.game.draw(draw_score=False,draw_hits=True)
            pygame.display.update()

            if game_info.left_score >= 1 or game_info.right_score >= 1 or game_info.left_hits >= 50:
                self.calculate_fitness(genome1,genome2,game_info)
                break

        # pygame.quit()

    def calculate_fitness(self,genome1,genome2,game_info):
        genome1.fitness += game_info.left_hits
        genome2.fitness += game_info.right_hits
         

def eval_genomes(genomes,config): #function that trains each genome (each genome will play against all other genomes in that population)
    width,height = 700,500
    window = pygame.display.set_mode((width,height))

    for i,(genome_id1,genome1) in enumerate(genomes): #first genome,second genome
        if i == len(genomes)-1: #if i == max len of genomes so if last genome then break out
            break
        genome1.fitness = 0 
        for genome_id2,genome2 in genomes[i+1:]: #second genome,third genome
            genome2.fitness = 0 if genome2.fitness == None else genome2.fitness

            game = PongGame(window,width,height)
            game.train_ai(genome1,genome2,config)


def run_neat(config):
    # p = neat.Checkpointer.restore_checkpoint("neat-checkpoint-27") #line to reload checkpoint replace this line with line 46 if there already is a checkpoint
    p = neat.Population(config) #creates a population size
    p.add_reporter(neat.StdOutReporter(True)) #returns data to the console about what generation is currently training,best fitness,...
    stats = neat.StatisticsReporter()
    p.add_reporter(stats)
    p.add_reporter(neat.Checkpointer(1)) #saves a checkpoint of genome data (fitness) in case have to re run script again (training can take hours or days depends on training for)
    
    winner = p.run(eval_genomes,5) #takes a fitness function and number of generations to run for ( will automatically pass (genomes,config) to fitness function ) winner returns the best genome at the end of 50 generations or the genome that meets the fitness threshold

    with open("best_ai.pickle",'wb') as p: #saves the best genome data of the max generations to a pickle file so can open later to have player vs best genome in max generations
        pickle.dump(winner,p)


def player_vs_ai(config): #loads best genome data to a python object to play against a player
    width,height = 700,500
    window = pygame.display.set_mode((width,height))
    with open("best_ai.pickle","rb") as p: 
        winner = pickle.load(p)

    game = PongGame(window,width,height)
    game.test_ai(winner,config)

if __name__ == "__main__":
    local_dir = os.path.dirname(__file__)
    config_path = os.path.join(local_dir,"config.txt")

    config = neat.Config(
        neat.DefaultGenome, neat.DefaultReproduction,
        neat.DefaultSpeciesSet, neat.DefaultStagnation,
        config_path
    ) #passes in the needed config settings

    ##run game ai vs ai
    # run_neat(config) #sets up background info for statistics on population

    ##run game player vs ai if 
    player_vs_ai(config)