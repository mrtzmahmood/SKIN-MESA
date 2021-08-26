import random
import math
from mesa import Agent
import numpy as np
import pandas as pd
from numpy import random as rnd
import random
import math
from functools import reduce
import inspect
import statistics
import itertools
seed = 7

np.random.seed(seed)
random.seed(seed)
class firmAgent(Agent):
    def __init__(self, unique_id, model):
        super().__init__(unique_id, model)
        self.ih = []            
        self.inputs = []
        self.research_direction = 'random'
        self.done_rad_research = False
        self.partners = []
        self.previous_partners = []        
        self.suppliers = []
        self.customers = []
        self.new_ih = True
        self.selling = False
        self.trading = False
        self.net = []
        self.hq = False
        self.age = 0       
        self.capabilities = np.array([])
        self.abilities = np.array([]) 
        self.expertises = np.array([])
        self.advert = []
        self.ability_to_research = 0
        self.product = 0
        self.quality = 0
        self.total_cost = 0
        self.price = 0
        self.sales = 0
        self.last_reward = 0
        self.capital = model.initial_capital
        self.pos = np.array((0, 0))
        self.breed = 'firm'
        
    def make_kene (self):    
        cap_capacity = np.log(1 + (self.capital / self.model.capital_knowledge_ratio))
        if cap_capacity < 5:
            cap_capacity = 5
        while len (self.capabilities) < cap_capacity:
            candidate_capability = rnd.randint(self.model.nCapabilities) + 1
            if candidate_capability not in self.capabilities :
                self.capabilities = np.insert(self.capabilities, 0, candidate_capability, axis=0)
                
        while len (self.abilities) < len(self.capabilities):
            self.abilities = np.insert(self.abilities, 0, rnd.uniform(10), axis=0)
            self.expertises = np.insert(self.expertises, 0, rnd.randint(10) + 1, axis=0)  
        
    def make_innovation_hypothesis (self): 
        self.ih = []
        location = 0
        kene_length = len(self.capabilities)
        ih_length = min ([rnd.randint(self.model.max_ih_length) + 2, kene_length ])
        while ih_length > 0:
            location = rnd.randint(kene_length) 
            if location not in self.ih:
                self.ih.append(location)
                ih_length -= 1
        self.ih.sort()
        self.research_direction = 'random'
        self.new_ih = True
        
    def make_advert (self):
        self.advert = self.capabilities[self.ih]

    def make_product (self):
        self.product = self.map_artefact(self.ih, self.model.raw_materials, self.model.nProducts)
        if self.product > self.model.end_products :
            self.price = self.model.final_price
        else:
            self.price = rnd.randint(self.model.maxPrice) + 1
            
    def make_quality (self):
        self.quality = reduce((lambda x,z: x +z), self.abilities[self.ih] * list(map(lambda y: 1-math.exp(-y), self.expertises[self.ih]))) % 10

    def map_artefact (self, locations, bottom, top):
        return  int (reduce((lambda x,y: x + y), self.capabilities[locations] * self.abilities[locations]) % (top - bottom)) + bottom

    def adjust_ih(self, location):
        elem = 0
        i = 0
        while i < len (self.ih):
            elem = self.ih[i]
            if elem > location:
                self.ih[i] = elem - 1
            i +=1
            
    def forget_capability(self, location):
        self.capabilities = np.delete(self.capabilities, location) 
        self.abilities = np.delete(self.abilities, location) 
        self.expertises = np.delete(self.expertises, location) 
        self.adjust_ih (location)       
        
    def make_inputs (self):
        self.inputs= []
        number_of_inputs = min ([len(self.ih), rnd.randint(self.model.nInputs) + 1])
        tries = 0
        while (len(self.inputs) < number_of_inputs) and (tries < 10*number_of_inputs):
            start_loc = rnd.randint(len(self.ih))
            end_loc = start_loc + rnd.randint(len(self.ih) - start_loc)
            input_art = self.map_artefact (self.ih[start_loc:(end_loc + 1)], 0, self.model.end_products)
            if (input_art != self.product) and (input_art not in self.inputs):
                self.inputs = [input_art] + self.inputs
            else:
                tries +=1
             
        if len(self.inputs) < number_of_inputs:
            self.inputs = [self.model.nProducts + 1]
        if self.product > self.model.end_products:
            raw_inputs_only = True
            for n in self.inputs:
                if n > self.model.raw_materials:
                    raw_inputs_only = False
            if raw_inputs_only:
                self.inputs = [self.model.nProducts + 1] 
                
    def adjust_expertise(self):
        if self.model.Adj_expertise:
            location = 0
            while location < len(self.capabilities):
                expertise= self.expertises [location]
                if location in self.ih:
                    if expertise < 10:
                        self.expertises [location] = expertise + 1
                else:
                    if expertise > 0:
                        self.expertises [location] = expertise - 1
                    else:
                        self.forget_capability(location)
                        location -=  1
                location += 1
                
    def manufacture (self):
        if self.new_ih:
            self.make_product()
            self.make_inputs()
            self.new_ih = False
        self.make_quality()
        self.adjust_expertise()
           
    def compatible (self, possible_partner):
        attractiveness = 0
        if not (possible_partner == self or possible_partner.hq):
            if not (self.net != [] and possible_partner in self.net.members):
                if not(self in possible_partner.partners or possible_partner in self.partners):
                    if self.model.partnership_strategy =='conservative':
                        attractiveness = len (self.intersection(self.advert, possible_partner.advert)) \
                                         / (min([len(self.advert), len(possible_partner.advert)]))
                    else:
                        if len (self.intersection(self.advert, possible_partner.advert)) >=1:
                            attractiveness = len(self.difference (self.advert, possible_partner.advert)) \
                                             / (len(self.advert) + len(possible_partner.advert))
                        else:
                            attractiveness = 0
                    return attractiveness > self.model.attractiveness_threshold
                else:
                    return False
            else:
                return False
        else:
            return False
    
    def intersection (self, set_a, set_b):
        set_c = list(set(set_a) & set(set_b))
        return set_c    
    
    def difference (self, set_a, set_b):
        set_c = list(set(set_a) & set(set_b))
        set_b = list(set(set_a) | set(set_a)) 
        set_b = list(set(set_b) - set(set_b))
        return set_b      

    def find_partners (self):
        candidates = []
        candidates = self.previous_partners + \
                     list(filter(lambda s: isinstance(s, firmAgent) , self.suppliers)) + \
                     list(filter(lambda c: isinstance(c, firmAgent) , self.customers))
        if len (candidates) < self.model.max_partners:
            xtra = min ([(self.model.max_partners - len(candidates)), len([a for a in self.model.schedule.agents if a.breed == 'firm'])])
            candidates = candidates + random.sample([a for a in self.model.schedule.agents if a.breed == 'firm'], xtra)
        if len (candidates) > self.model.max_partners:
            candidates = candidates + random.sample([c for c in candidates if c.breed == 'firm'], self.model.max_partners)
        myself = self                     
        candidates = [c for c in candidates if c.compatible(myself)]
        self.partners = self.partners + candidates
        for c in candidates:
            c.partners = [myself] + c.partners
        
    def do_research (self):
        if self.last_reward < self.model.success_threshold:
            if self.capital <= self.model.low_capital_threshold:
                self.do_radical_research()
            else:
                self.do_incremental_research()

    def do_radical_research (self):
        if self.model.Rad_research:
            self.done_rad_research = True
            capability_to_mutate =  rnd.randint(len(self.capabilities))
            new_capability =  rnd.randint(self.model.nCapabilities) + 1
            while new_capability in self.capabilities:
                new_capability =  rnd.randint(self.model.nCapabilities) + 1
            self.capabilities[capability_to_mutate] = new_capability
            self.new_ih = True
            self.pay_tax (self.model.radical_research_tax)
            
    def do_incremental_research (self):
        if self.model.Incr_research:
            if self.research_direction =='random':
                self.ability_to_research = rnd.randint(len(self.ih))
                if rnd.randint(2) == 1:
                    self.research_direction = 'up'
                else:
                    self.research_direction = 'down'        
            new_ability = self.abilities [self.ability_to_research]
            if self.research_direction == 'up':
                new_ability = new_ability + (new_ability / self.capabilities [self.ability_to_research])
            else:
                new_ability = new_ability - (new_ability / self.capabilities [self.ability_to_research]) 
                
            if new_ability <=0:
                new_ability = 0
                self.research_direction = 'random'
            if new_ability > 10:
                new_ability = 10
                self.research_direction = 'random'
            self.abilities [self.ability_to_research] = new_ability
            self.new_ih = True
            self.pay_tax (self.model.incr_research_tax)
            
    def learn_from_partners (self):
        myself = self
        for p in self.partners:
            p.merge_capabilities (myself)
        self.make_innovation_hypothesis()
        
    def merge_capabilities (self, other_firm):
        self.add_capabilities(other_firm)
        myself = self
        other_firm.add_capabilities (myself)
        
    def add_capabilities (self, other_firm):
        my_position = 0
        for ih in other_firm.ih:
            capability =  other_firm.capabilities[ih]
            if capability in self.capabilities:
                result = np.where(self.capabilities == capability)
                my_position = result[0]
                if self.expertises[my_position] < other_firm.expertises[ih]:
                    self.expertises [my_position] = other_firm.expertises[ih]
                    self.abilities [my_position] = other_firm.abilities[ih]
            else:
                if len(self.capabilities) < ((self.capital / self.model.capital_knowledge_ratio) + 1):
                    self.capabilities = np.concatenate((self.capabilities, capability), axis=None)
                    self.abilities = np.concatenate((self.abilities, other_firm.abilities[ih]), axis=None) 
                    other_expertise = other_firm.expertises[ih] -1
                    if other_expertise < 2:
                        other_expertise = 2
                    self.expertises = np.concatenate((self.expertises, other_expertise), axis=None) 
                    
    def pay_taxes (self):
        self.pay_tax(self.model.depreciation)
    
    def pay_tax (self, amount):
        self.capital = self.capital - amount
    
    def collaborate (self):
        if self.model.Partnering and not self.trading :
            self.find_partners()
            if len(self.partners) > 0:
                self.learn_from_partners()
                self.pay_tax (self.model.collaboration_tax*(len(self.partners)))
                
    def take_profit (self):
        self.last_reward = 0
        if self.trading :
            self.last_reward = len(self.customers) * (self.price - self.total_cost)
            
    def adjust_price (self):
        if self.model.Adj_price:
            if self.trading:
                if len(self.customers) > 4:
                    self.price = self.price * 1.1
                if len(self.customers) == 0:
                    self.price = self.price * 0.9
                
    def product_desired (self, market):
        my_product = self.product
        if my_product > self.model.end_products:
            return True
        founded_firms = [m for m in market if my_product in m.inputs]
        return (len(founded_firms) > 0)
    
    def inputs_available (self, market):
        for i in self.inputs:
            founded_firms = [m for m in market if m.product == i]           
            if i >= self.model.raw_materials and len(founded_firms) ==0:
                return False
        return True
                
    def profitable  (self, market):
        if not self.product_desired (market):
            return False
        if not self.inputs_available (market):
            return False
        if not self.trading:
            self.suppliers = []
            for i in self.inputs:
                supplier = []
                if i >= self.model.raw_materials:
                    possible_suppliers = [m for m in market if m.product == i]
                    if possible_suppliers:                       
                        min_price = min([ps.price for ps in possible_suppliers])
                        cheapest_suppliers = [cs for cs in possible_suppliers if cs.price== min_price]                        
                        max_quality =max([cs.quality for cs in cheapest_suppliers])
                        supplier = [random.choice([s for s in cheapest_suppliers if s.quality == max_quality])]
                else:
                    supplier=['raw-material']
                self.suppliers = supplier + self.suppliers  
        return (self.price > self.cost_price())
    
    def cost_price (self):
        total = 0
        for s in self.suppliers:
            if s == 'raw-material':
                total += self.model.raw_cost 
            else:
                total += s.price
        return total

    def purchase (self):
        self.total_cost = 0
        for s in self.suppliers:
            if s=='raw-material':
                self.total_cost += self.model.raw_cost
            else:
                self.total_cost += s.price
                myself = self
                s.capital += s.price
                s.customers = [myself] + s.customers
                s.sales += s.price
        self.capital += self.total_cost
        if self.product > self.model.end_products:
            self.capital += self.model.final_price
            self.customers = ['end-user'] + self.customers
            
    def do_admin (self):
        self.previous_partners = self.previous_partners + self.partners
        self.age +=1
        if self.capital < 0:
            self.exit()
            
    def exit (self):
        # Remove from net
#        for a in model.schedule.agents:  
#            if a.breed == 'network':
        myself = self
        firms = [agent for agent in self.model.schedule.agents if agent.breed == 'firm']
        for f in firms:  
            if myself in f.partners: f.partners.remove(myself)
            if myself in f.previous_partners: f.previous_partners.remove(myself)
        selected_firms = [f for f in firms if myself in f.suppliers or myself in f.customers]
        for sf in selected_firms:
            if myself in sf.customers: sf.customers.remove(myself)
            if myself in sf.suppliers: sf.suppliers.remove(myself)
            sf.trading = False
        self.model.schedule.remove(self)
        
    def clone_kene (self, firm_to_clone):
        ih_pos = 0
        for i in range(int(min ([len(firm_to_clone.ih), (np.log((self.capital / self.model.capital_knowledge_ratio)+1)+1)]))):
            triple_pos = firm_to_clone.ih[ih_pos]
            self.capabilities = np.insert(self.capabilities, 0, firm_to_clone.capabilities[triple_pos], axis=0)
            self.abilities = np.insert(self.abilities, 0, firm_to_clone.abilities[triple_pos], axis=0)
            self.expertises = np.insert(self.expertises, 0, firm_to_clone.expertises[triple_pos], axis=0)
            ih_pos += 1 
    
    def step(self):
        if self.model.partnership_strategy != 'no partner':
            self.collaborate()
        self.do_research()
        self.manufacture()
        self.pay_taxes()
        
    @staticmethod        
    def rpt_partners (a):
        partners = [len(agent.partners) for agent in a.model.schedule.agents if agent.breed == 'firm' and agent.partners]
        return partners
    
    @staticmethod        
    def rpt_age_distribution (a):
        age_list = [agent.age for agent in a.model.schedule.agents if agent.breed == 'firm']
        if max(age_list) > 0:
            return age_list 
        else:
            return [] 

    @staticmethod        
    def rpt_size_distribution (a):
        cap_list = [agent.capital for agent in a.model.schedule.agents if agent.breed == 'firm']
        max_cap = max(cap_list)
        return [100*c/max_cap for c in cap_list] 
        
    @staticmethod        
    def rpt_product (a):
        return [agent.product for agent in a.model.schedule.agents if agent.breed == 'firm']
    
    @staticmethod        
    def rpt_inputs (a):
        inputs_lists = [agent.inputs for agent in a.model.schedule.agents if agent.breed == 'firm' and agent.inputs]
        inputs = list(itertools.chain(*inputs_lists))
        return inputs           

class networkAgent(Agent):
    def __init__(self, unique_id, model):
        super().__init__(unique_id, model)
        self.members = []            
        self.hq = []
        self.pos = np.array((0, 0))
        self.breed = 'network'
        
#    def step(self):
    
#        if self.model.communication_regime == "DW" and self.model.original:
#            other = self.random.choice(self.model.schedule.agents)
#        else:
#            myself = self
#            self.update_opinion(myself)

