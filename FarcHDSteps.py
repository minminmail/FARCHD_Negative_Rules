from Apriori import Apriori
from DataBase import DataBase
from Populate import Populate
from RuleBase import RuleBase


class FarcHDSteps:
    data_base = None
    rule_base = None
    train_mydataset = None

    k_parameter = None
    type_inference = None

    minsup = None

    minconf = None

    depth = None

    seed_int = None

    population_size = None
    bits_gen = None
    max_trials = None
    alpha = None

    def __init__(self, nlabels, train_mydataset, k_parameter, type_inference, minsup, minconf, depth, alpha,max_trials):
        self.class_value = 0
        # print("__init__ of data_row")
        self.nlabels = nlabels
        self.train_mydataset = train_mydataset
        self.k_parameter = k_parameter
        self.type_inference = type_inference
        self.minsup = minsup
        self.minconf = minconf
        self.depth = depth
        self.alpha = alpha
        self.max_trials= max_trials

        self.data_base = DataBase()
        self.data_base.init_with_three_parameters(self.nlabels, self.train_mydataset)
        self.rule_base = RuleBase()
        self.rule_base.init_with_five_parameters(self.data_base, self.train_mydataset, self.k_parameter,
                                                 self.type_inference)
        self.apriori = Apriori()
        self.apriori.multiple_init(self.rule_base, self.data_base, self.train_mydataset, self.minsup, self.minconf,
                                   self.depth)

    def execute_FarcHd(self, seed_int, population_size, bits_gen):
        self.seed_int = seed_int
        self.population_size = population_size
        self.bits_gen = bits_gen

        self.apriori.generate_rb()
        self.rules_stage1 = self.apriori.get_rules_stage1()
        self.rules_stage2 = self.rule_base.get_size()
        print("self.rules_stage1")
        print(self.rules_stage1)
        print("self.rules_stage2")
        print(self.rules_stage2)

        self.pop = Populate()

        self.pop.init_with_multiple_parameters(self.seed_int, self.train_mydataset, self.data_base, self.rule_base,
                                               self.population_size, self.bits_gen, self.max_trials, self.alpha)
        self.pop.generation()

        print("Building classifier")
        self.rule_base = self.pop.get_best_RB()

        self.rules_stage3 = int(self.rule_base.get_size())
        print("Granularity rule number is  rules_stage3" +str(self.rules_stage3))