from Individual import Individual
from DataBase import DataBase
from RuleBase import RuleBase
from MyDataSet import MyDataSet
from random import randrange, randint
import logging
import random
from Logger import Logger 


class Populate:
    population_array = []

    alpha = None
    w1 = None
    l_value = None
    lini = None

    n_variables = None
    pop_size = None
    maxtrials = None
    ntrials = None
    bits_gen = None
    best_fitness = None
    best_accuracy = None
    selected_array = []

    train_mydataset = None
    data_base = None
    rule_base = None
    logger = None
    seed_value =None

    """
    * Maximization
    * @ param a int first number
    * @ param b int second number
    """

    def better(self, value_a, value_b):
        if value_a > value_b:
            return True
        else:
            return False

    """"
    None attribute will be initialized.
    """

    def __init__(self):
        self.logger = Logger.set_logger()
        pass

    """
    * @param train Training dataset
    * @param dataBase Data Base
    * @param ruleBase Rule set
    * @param size Population size
    * @param BITS_GEN Bits per gen
    * @param maxTrials Maximum number of evaluacions
    * @param alpha Parameter alpha
    
    """

    def init_with_multiple_parameters(self, seed_value,train_mydataset_pass, data_base, rule_base_pass, size, bits_gen, maxtrials,
                                      alpha):
        self.seed_value = seed_value
        self.logger=Logger.set_logger()
        self.data_base = data_base
        self.train_mydataset = train_mydataset_pass
        self.rule_base = rule_base_pass
        self.bits_gen = bits_gen

        self.n_variables = data_base.num_variables()
        self.pop_size = size
        self.alpha = alpha
        self.maxtrials = maxtrials
        self.lini = ((data_base.get_nlabels_real() * bits_gen) + rule_base_pass.get_size()) / 4.0
        self.l_value = self.lini
        rule_size = rule_base_pass.get_size()
        self.w1 = self.alpha * rule_size

        self.population_array = []
        self.selected_array = [0 for x in range(self.pop_size)]

        """
        * Run the CHC algorithm (Stage 3) 
        """

    def generation(self):
        self.init()
        self.evaluate(0)

        while True:
            self.selection()
            self.cross_over()
            self.evaluate(self.pop_size)
            self.elitist()
            if not self.has_new():
                self.l_value = self.l_value - 1
                if self.l_value < 0.0:
                    self.restart()

            if self.ntrials >= self.maxtrials:
                break

    def init(self):
        self.logger=Logger.set_logger()
        ind = Individual()
        ind.init_with_parameter(self.rule_base, self.data_base, self.w1)
        ind.reset()
        self.population_array.append(ind)
        for i in range(1, self.pop_size):
            ind = Individual()
            ind.init_with_parameter(self.rule_base, self.data_base, self.w1)
            ind.random_values(self.seed_value)
            self.population_array.append(ind)
            print(" the init loop  method added "+ str(i)+"individuals ")

        self.best_fitness = 0.0
        self.ntrials = 0

    def evaluate(self, pos):
        for i in range(pos, len(self.population_array)):
            self.population_array[i].evaluate()
        self.ntrials = self.ntrials + (len(self.population_array) - pos)

    def selection(self):

        aux = None
        random_value = None

        for i in range(0, self.pop_size):
            self.selected_array[i] = i

        for i in range(0, self.pop_size):
            """
            numpy.random.randint(low, high=None, size=None, dtype='l')¶
            Return random integers from low (inclusive) to high (inclusive). looks include the high also, 
            """
            # if we add the seed here, then reduce the prediction rate
            random.seed(self.seed_value)
            random_value = randint(0, self.pop_size-1)
            #print("random is :" + str(random_value))
            aux = self.selected_array[random_value]
            self.selected_array[random_value] = self.selected_array[i]
            self.selected_array[i] = aux
        """ 
        for i in range(0, self.pop_size):
            self.logger.debug( "In selection, self.selected_array["+ str(i)+ "]"+ str(self.selected_array[i]))
        """

    def xpc_blx(self, d_value, son1_individual, son2_individual):
        son1_individual.xpc_blx(son2_individual, d_value,self.seed_value)

    def hux(self, son1_individual, son2_individual):
        son1_individual.hux(self.seed_value,son2_individual)

    def cross_over(self):

        dist = None
        dad_individual = None
        mom_individual = None
        son1_individual = None
        son2_individual = None

        for i in range(0, self.pop_size, 2):
            dad_individual = self.population_array[self.selected_array[i]]
            mom_individual = self.population_array[self.selected_array[i + 1]]
            dist = float(dad_individual.dist_hamming(mom_individual, self.bits_gen))
            dist = dist / 2.0

            if dist > self.l_value:
                son1_individual = dad_individual.clone()
                son2_individual = mom_individual.clone()

                self.xpc_blx(1.0, son1_individual, son2_individual)
                self.hux(son1_individual, son2_individual)

                son1_individual.on_new()
                son2_individual.on_new()

                self.population_array.append(son1_individual)
                self.population_array.append(son2_individual)
        """  
        for i in range (0, len(self.population_array)):
            self.logger.debug("In cross_over, self.population_array[" + str(i)+ "].fitness" + str(self.population_array[i].fitness))
            """

    def elitist(self):
        
        # need to know which order to sort ,how to sort, if the sort will be saved
        self.population_array.sort(key=lambda x: x.fitness, reverse=True)
        """
        for i in range (0, len(self.population_array)):
            self.logger.debug("In elitist, before pop, self.population_array[" + str(i)+ "].fitness" + str(self.population_array[i].fitness))
            """
        
        while len(self.population_array) > self.pop_size:
            # print("len(self.population_array)"+str(len(self.population_array)))
            # print("len(self.pop_size)" + str(self.pop_size))
            # print("value " + str(self.population_array[self.pop_size]))
            # self.logger.debug("In elitist,  pop， self.population_array[" + str(self.pop_size)+ "].fitness" + str(self.population_array[self.pop_size].fitness))
            self.population_array.pop(self.pop_size)
        """
        for i in range (0, len(self.population_array)):
            self.logger.debug("In elitist, after pop, self.population_array[" + str(i)+ "].fitness" + str(self.population_array[i].fitness))
        """

        self.best_fitness = self.population_array[0].get_fitness()
        # self.logger.debug("In elitist of Populate class, self.best_fitness  " + str(self.best_fitness))

        #print("in elitist in population class the best_fitness is :" +str(self.best_fitness))

    def has_new(self):

        state = None
        ind = None
        state = False

        for i in range(0, self.pop_size):
            ind = self.population_array[i]
            if ind.is_new():
                ind.off_new()
                state = True

        return state

    def restart(self):

        i = None
        dist = None
        ind = None
        self.w1 = 0.0

        self.population_array.sort(key=lambda x: x.fitness,reverse=True)

        ind = self.population_array[0].clone()
        # self.logger.debug("in restart , Populate class, ind.fitness, self.population_array[0] is :  " + str(ind.fitness))
        print(" in restart the selected self.population_array[0] fitness is "+str(ind.fitness))
        ind.set_w1_value(self.w1)

        self.population_array.clear()
        self.population_array.append(ind)

        for i in range(1, self.pop_size):
            ind = Individual()
            ind.init_with_parameter(self.rule_base, self.data_base, self.w1)
            ind.random_values(self.seed_value)
            self.population_array.append(ind)

        self.evaluate(0)

        """
        for i in range ( 0, len(self.population_array)):
           
            self.logger.debug("in restart of populate class, after  self.evaluate(0), self.population_array["+str(i)+"].fitness"+ str(self.population_array[i].fitness))
        """

        self.l_value = self.lini

        """
        * Return the best individual in the population 
        """

    def get_best_RB(self):

        # for i in range ( 0, len(self.population_array)):
            # print("self.population_array["+str(i)+"].fitness")
            # self.logger.debug("in get_best_RB of populate, in populate class, self.population_array["+str(i)+"].fitness"+ str(self.population_array[i].fitness))
            # print(self.population_array[i].fitness)


        self.population_array.sort(key=lambda x: x.fitness,reverse=True)
        rule_base = self.population_array[0].generate_rb() 
        # self.logger.debug("in get_best_RB , in populate class, return rule_base self.population_array["+str(0)+"].fitness"+ str(self.population_array[0].fitness))

        return rule_base
