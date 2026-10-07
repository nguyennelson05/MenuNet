import Agent
import Prompts
import os
import torch
import Model.Train as MenuNet
import random
import numpy as np

class Auction:
    def __init__(self, round = 1, distribution = [1,1]):
        self.api_key = "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"
        self.rounds = round
        self.dist = distribution
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

    def calculate_profit(self, valuations, allocation, payment):
        expected_value = 0
        for i in range(len(valuations)):
            expected_value += valuations[i] * allocation[i]
        profit = expected_value-payment
        #print("VALUATION, ALLOCATION  ", valuations, allocation)
        #print("EXPECTED VALUE, PAYMENT  ", expected_value, payment)
        #print("PROFIT  ", profit)
        return float(profit)

    def calculate_regret(self, menu, bids, valuations, options):
        alloc, payment = menu(bids)
        alloc_true, payment_true = menu(valuations)
        option = options.get(payment)
        option_true = options.get(payment_true)
        if option == option_true:
            regret = 0
        else:
            bids_profit = self.calculate_profit(valuations, alloc, payment)
            valuation_profit = self.calculate_profit(valuations, alloc_true, payment_true)
            regret = round(bids_profit - valuation_profit, 4)
            #print(f"True Option: {option_true}")
        return regret, option


    #Default Settings: No history, different valuations (bidders), same items (vi_bar)
    def auction(self, rule: bool, hist: bool, same_bidder: bool, diff_items: bool, model):

        llm_bidder = Agent.LLMBidder(api_key=self.api_key, model = model)
        truth_counter = 0
        regret = 0
        round_history = []
        if same_bidder:
            valuations_tensor = llm_bidder.make_valuation(self.dist)
            valuations = [round(float(x), 4) for x in valuations_tensor]

        if not diff_items:
            menu = Utilities.load_model(self.dist, self.device)
            X, p = menu.mech()
            menu_dict = Utilities.print_menu_options(X.detach(), p)
            menu_option_counter = {key: 0 for key in menu_dict}
            payment_to_menu_option = {payment: key for key, (alloc,payment) in menu_dict.items()}
        else:
            menu_option_counter = {}
        
        #print("\n---AUCTION STARTS---")
        with torch.no_grad():
            for curr_round in range(1, self.rounds+1):
                if diff_items:
                    vi_bar = random.choice([1, 1.5, 2, 2.5])
                    self.dist = [1,vi_bar]
                    menu = Utilities.load_model(self.dist, self.device)
                    X, p = menu.mech()
                    menu_dict = Utilities.print_menu_options(X.detach(), p)
                    payment_to_menu_option = {payment: key for key, (alloc,payment) in menu_dict.items()}
        
                if not same_bidder:
                    valuations_tensor = llm_bidder.make_valuation(self.dist)
                    valuations = [round(float(x), 4) for x in valuations_tensor]
                instruction = Prompts.game_instruction("MenuNet", valuations, self.dist, 
                                                    if_history=hist, history = round_history,
                                                    if_rule=rule,)

                bids = llm_bidder.make_bid(instruction).strip("()").split(",")
                bids = [round(float(x), 4) for x in bids]
                llm_bids = torch.tensor(bids, device=self.device, dtype=torch.float32)

                if bids == valuations:
                    truth_counter += 1 
                curr_regret, option = self.calculate_regret(menu, llm_bids, valuations_tensor, payment_to_menu_option)
                regret += curr_regret
                if not diff_items:
                    menu_option_counter[option] += 1
    
                if hist:
                    options_str = ", ".join(
                        f"{key}: Allocations = {allocs} & payment = {payment:.4f}"
                        for key, (allocs, payment) in menu_dict.items())
                    round_history.append(
                        f"Round {curr_round}: The previous bids, {bids[0]} for the apple and {bids[1]} for the banana, "
                        f"were made from these valuations domains:[0,{self.dist[0]}] and [0,{self.dist[1]}]."
                        f"This returned {option} from these possible menu options: {options_str}.\n")

                if curr_round%10 == 0 or curr_round == 1:
                    print(f"ROUND {curr_round}:")
                    print(f"  Valuations: {valuations} ") 
                    print(f"  Bids: {bids} ")
                    if bids != valuations:
                        print("  UNTRUTHFUL BID")
                    if option:
                        print(f"  {option}")

        #Utilities.print_counters(truth_counter, menu_option_counter)
        return truth_counter, regret, menu_option_counter


#Utilities for Auction
class Utilities():    

    @staticmethod
    def load_model(dist, device):
        model_name = f"[0,{dist[0]}] x [0,{dist[1]}]"
        model_path = os.path.join(os.path.dirname(__file__), f"Model/models/{model_name}.pt")
        print(f"Auction for Valuation Distribution: {model_name}")
        params = dist, 2, 3
        menu = MenuNet.load_trained_model(params, model_path, device)
        return menu

    @staticmethod
    def print_menu_options(X,p):
        menu_dict = {}
        #print("All Menu Options")
        k = len(X[0])
        for j in range(k):
            alloc = [round(float(a), 4) for a in X[:, j]]
            payment = float(p[j].item())
            label = " (exit)" if j == k - 1 else ""
            menu_dict[f"Menu Option {j+1}{label}"] = [alloc, payment]
            #print(f"  Menu Option {j+1}{label}: [{alloc}, {payment:.4f}]")
        return menu_dict       
    
