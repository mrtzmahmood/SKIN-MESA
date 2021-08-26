from mesa import Model
from mesa.time import RandomActivation
from mesa.datacollection import DataCollector
from mesa.space import ContinuousSpace
from .agent import firmAgent, networkAgent
import numpy as np
from numpy import random as rnd
import random
import math
import statistics
seed = 7
np.random.seed(seed)

class skinModel(Model):
    def __init__ (self, nFirms, nProducts, nInputs, n_big_firms_percent, reward_to_trigger_start_up, attractiveness_threshold,
                       partnership_strategy, success_threshold, in_out_product_percent, initial_capital, 
                       Partnering, Networking, Start_ups, Adj_expertise, Adj_price,Incr_research, Rad_research, width, height, max_iters, batch_mode):       
        self.nFirms = nFirms
        self.nProducts = nProducts
        self.nInputs = nInputs
        self.n_big_firms_percent = n_big_firms_percent
        self.reward_to_trigger_start_up = reward_to_trigger_start_up
        self.attractiveness_threshold = attractiveness_threshold
        self.partnership_strategy = partnership_strategy
        self.success_threshold = success_threshold
        self.in_out_product_percent = in_out_product_percent
        self.initial_capital = initial_capital
        self.Partnering= Partnering
        self.Networking = Networking
        self.Start_ups = Start_ups
        self.Adj_expertise= Adj_expertise
        self.Adj_price = Adj_price
        self.Incr_research = Incr_research
        self.Rad_research = Rad_research       
        self.initial_capital_for_big_firms = initial_capital * 10
        self.maxPrice = 1000
        self.max_ih_length = 9
        self.nCapabilities = 1000
        self.low_capital_threshold = 1000
        self.capital_knowledge_ratio = 20
        self.incr_research_tax = 100
        self.radical_research_tax = 100
        self.collaboration_tax = 100
        self.depreciation= 100
        self.raw_cost = 1
        self.final_price = 10 * self.maxPrice
        self.max_partners = 5
        self.raw_materials = math.floor((self.nProducts * self.in_out_product_percent) / 100)
        self.end_products = math.ceil((self.nProducts * (100 - self.in_out_product_percent)) / 100)
        self.space = ContinuousSpace(width, height, True, 0, 0)
        self.schedule = RandomActivation(self)
        self.width = width
        self.height = height
        self.max_iters = max_iters
        self.batch_mode = batch_mode
        self.iteration = 0  
        self.running = True
        model_reporters = {
            "Capital": lambda m: self.rpt_capital_mean (m),
            'In partnership': lambda m: self.rpt_in_partnership (m),
            'In network': lambda m: self.rpt_in_network (m),
            'Firms': lambda m: self.rpt_firms (m),
            "Networks": lambda m: self.rpt_networks (m),     
            'Successes': lambda m: self.rpt_succeeded_firms (m),
            'Start-ups': lambda m: self.rpt_startups (m),
            'Firms selling': lambda m: self.rpt_selling_firms (m),
            'Firms trading': lambda m: self.rpt_trading_firms (m),
            'Sales': lambda m: self.rpt_sales (m),
            'Profit': lambda m: self.rpt_profit (m),
            'Rate of radical research': lambda m: self.rpt_rate_of_radical_distribution(m)          
        }        
        agent_reporters = {
            'x': lambda a: a.pos[0],
            'y': lambda a: a.pos[1],
            'Partners': lambda a: a.rpt_partners (a),
            'Age distribution': lambda a: a.rpt_age_distribution(a),
            'Size distribution': lambda a: a.rpt_size_distribution(a),            
            'Product': lambda a: a.rpt_product(a),
            'Inputs': lambda a: a.rpt_inputs(a)
        }
        self.dc = DataCollector(model_reporters=model_reporters,
                                agent_reporters=agent_reporters)                
        # Create agents
        for i in range(self.nFirms):
            x = rnd.randint(self.width)
            y = rnd.randint(self.height)
            pos = np.array((x, y))
            fa = firmAgent(i, self)
            fa.pos = pos
            self.space.place_agent(fa, fa.pos)
            self.schedule.add(fa)
        firms = [agent for agent in self.schedule.agents if agent.breed == 'firm']
        big_firms = random.sample(firms, int((self.n_big_firms_percent * self.nFirms) / 100))
        
        for bf in big_firms:
            bf.capital = self.initial_capital_for_big_firms
            
        for f in firms:
            f.make_kene()
            f.make_innovation_hypothesis()
            f.make_advert()
            f.manufacture()
            
    def step(self):
            self.schedule.step()
            self.find_suppliers()
            self.buy()
            firms = [agent for agent in self.schedule.agents if agent.breed == 'firm']
            for f in firms:
                f.take_profit()
                f.adjust_price()
            self.create_start_ups()
            self.dc.collect(self)
        #   model.create_nets()
        #   networks = [agent for agent in self.schedule.agents if agent.breed == 'network']
        #   for a in networks:
        #       a.distribute_network_profits ()               
            firms = [agent for agent in self.schedule.agents if agent.breed == 'firm']
            for f in firms:
                f.do_admin()
            self.iteration += 1
            if not self.batch_mode and self.iteration > self.max_iters:
                self.running = False
        
    def find_suppliers(self):
        firms = [agent for agent in self.schedule.agents if agent.breed == 'firm']
        for f in firms:
            f.selling = False
        possible_firms = [f for f in firms if f.inputs_available(firms)]              
        previous_possible_firms =[]  
        while possible_firms != previous_possible_firms:
            previous_possible_firms = possible_firms
            possible_firms = [pf for pf in possible_firms if pf.product_desired (possible_firms)]
            possible_firms = [pf for pf in possible_firms if pf.inputs_available (possible_firms)]
        for pf in possible_firms:
            pf.selling = True   
        previous_possible_firms =[]      
        while possible_firms != previous_possible_firms:
            previous_possible_firms = possible_firms
            possible_firms = [pf for pf in possible_firms if pf.profitable (possible_firms)]
        for f in firms:
            f.trading = False
        for pf in possible_firms:
            pf.trading = True
        
    def buy (self):
        firms = [agent for agent in self.schedule.agents if agent.breed == 'firm']
        for f in firms:
            f.sales = 0
            f.customers = []
        for f in firms:
            if f.trading:
                f.purchase()
                
    def create_start_ups (self):
        firms = [agent for agent in self.schedule.agents if agent.breed == 'firm']
        biggest_reward = max(f.last_reward for f in firms)
        i= len (firms)
        if i > 0 and self.Start_ups:
            if biggest_reward > self.reward_to_trigger_start_up:
                for i in range(int (math.log10(biggest_reward))):
                    self.make_start_up()

    def make_start_up (self):
        firms = [agent for agent in self.schedule.agents if agent.breed == 'firm']
        max_reward =  max(f.last_reward for f in firms)
        max_reward_firm = random.choice([f for f in firms if f.last_reward == max_reward])
        i= len (firms) + 1
        x = rnd.randint(self.width)
        y = rnd.randint(self.height)
        pos = np.array((x, y))
        fa = firmAgent(i, self)
        fa.pos = pos
        self.space.place_agent(fa, fa.pos)
        self.schedule.add(fa)
        fa.capital = self.initial_capital
        fa.breed = 'firm'
        fa.clone_kene(max_reward_firm)
        fa.make_innovation_hypothesis()
        fa.make_advert()
        fa.manufacture()
        
    @staticmethod        
    def rpt_networks (model):
        return len([agent for agent in model.schedule.agents if agent.breed == 'network'])

    @staticmethod        
    def rpt_firms (model):
        return len([agent for agent in model.schedule.agents if agent.breed == 'firm'])
    
    @staticmethod        
    def rpt_capital_mean (model):
        capitals = [math.log10(agent.capital) for agent in model.schedule.agents if agent.breed == 'firm' and agent.capital > 0]
        capitals_mean  = 0
        if len(capitals) > 0:
            capitals_mean = statistics.mean(capitals) 
        return capitals_mean

    @staticmethod        
    def rpt_in_partnership (model):
        partners = len([agent for agent in model.schedule.agents if agent.breed == 'firm' and agent.partners])
        firms = len([agent for agent in model.schedule.agents if agent.breed == 'firm'])
        return 100*partners/firms

    @staticmethod        
    def rpt_in_network (model):
        networks = len([agent for agent in model.schedule.agents if agent.breed == 'firm' and agent.net])
        firms = len([agent for agent in model.schedule.agents if agent.breed == 'firm'])
        return 100*networks/firms

    @staticmethod        
    def rpt_selling_firms (model):
        selling_firms = len([agent for agent in model.schedule.agents if agent.breed == 'firm' and agent.selling])
        firms = len([agent for agent in model.schedule.agents if agent.breed == 'firm'])
        return 100*selling_firms/firms
    @staticmethod        
    def rpt_trading_firms (model):
        trading_firms = len([agent for agent in model.schedule.agents if agent.breed == 'firm' and agent.trading])
        firms = len([agent for agent in model.schedule.agents if agent.breed == 'firm'])
        return 100*trading_firms/firms   
    
    @staticmethod        
    def rpt_succeeded_firms (model):
        selling_firms = len([agent for agent in model.schedule.agents if agent.breed == 'firm' and agent.selling])
        succeeded_firms = len([agent for agent in model.schedule.agents if agent.breed == 'firm' and agent.last_reward > model.success_threshold])
        success = 0
        if selling_firms >0:
            success = 100 * succeeded_firms / selling_firms
        return success

    @staticmethod        
    def rpt_startups (model):       
        startups = len([agent for agent in model.schedule.agents if agent.breed == 'firm' and agent.age == 0 and model.iteration !=0 and not agent.hq])
        firms = len([agent for agent in model.schedule.agents if agent.breed == 'firm'])
        return 100*startups/firms 

    @staticmethod        
    def rpt_sales (model):
        sales = [agent.sales for agent in model.schedule.agents if agent.breed == 'firm' and agent.sales > 0]
        sales_mean  = 0
        if len(sales) > 0:
            sales_mean = statistics.mean(sales) 
        return sales_mean
    
    @staticmethod        
    def rpt_profit (model):
        profits = [agent.last_reward for agent in model.schedule.agents if agent.breed == 'firm' and agent.last_reward > 0]
        profits_mean  = 0
        if len(profits) > 0:
            profits_mean = statistics.mean(profits) 
        return profits_mean

    @staticmethod        
    def rpt_rate_of_radical_distribution (model):
        return len ([agent for agent in model.schedule.agents if agent.breed == 'firm' and agent.done_rad_research])
 

