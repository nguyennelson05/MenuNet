import Agent
import Prompts
import os
import torch
import Model.Train as MenuNet
import random
import numpy as np

class Auction:
    def __init__(self, round = 1, distribution = [1,1], api_key: str | None = None):
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        if not self.api_key:
            raise ValueError("Missing OpenAI API key. Set OPENAI_API_KEY or pass api_key=...")
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

    def calculate_regret_batched(self, menu, bids, valuations, options):
        try:
            batched = torch.stack([bids, valuations], dim=0)
            allocs, payments = menu(batched)
            alloc, alloc_true = allocs[0], allocs[1]
            payment = float(payments[0].item()) if torch.is_tensor(payments[0]) else float(payments[0])
            payment_true = float(payments[1].item()) if torch.is_tensor(payments[1]) else float(payments[1])
        except Exception:
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
    def auction(
        self,
        rule: bool,
        hist: bool,
        same_bidder: bool,
        diff_items: bool,
        model,
        reasoning,
        *,
        auction_type: str = "MenuNet",
        truncate_history: bool = True,
        history_limit: int | None = None,
        verbose: bool = True,
    ):
        llm_bidder = Agent.LLMBidder(api_key=self.api_key, model = model, reasoning=reasoning)
        truth_counter = 0
        regret = 0
        round_history = []
        menu_cache = {}
        options_history = []
        options_history_keys = set()
        if same_bidder:
            valuations_tensor = llm_bidder.make_valuation(self.dist)
            valuations = [round(float(x), 4) for x in valuations_tensor]
        if not diff_items:
            menu, menu_dict, payment_to_menu_option = Utilities.get_menu(self.dist, menu_cache, self.device)

        with torch.inference_mode():
            if not hist:
                round_inputs = []
                for curr_round in range(1, self.rounds+1):
                    if diff_items:
                        vi_bar = random.choice([1, 1.5, 2, 2.5])
                        self.dist = [1, vi_bar]
                    if not same_bidder:
                        valuations_tensor = llm_bidder.make_valuation(self.dist)
                        valuations = [round(float(x), 4) for x in valuations_tensor]

                    instruction = Prompts.game_instruction(auction_type, valuations, self.dist, if_history=hist,
                                                            history = round_history, if_rule=rule,)
                    
                    round_inputs.append({"round": curr_round, "dist": self.dist, "valuations_tensor": valuations_tensor,
                                          "valuations": valuations, "instruction": instruction,})

                bids_list = llm_bidder.make_bids([r["instruction"] for r in round_inputs])
                if len(bids_list) != len(round_inputs):
                    bids_list = [("", "")] * len(round_inputs)

                for idx, info in enumerate(round_inputs):
                    bid_text, reasoning_text = bids_list[idx] if idx < len(bids_list) else ("", "")
                    if not bid_text:
                        bid_text, reasoning_text = llm_bidder.make_bid(info["instruction"])

                    bids = bid_text.strip("()").split(",")
                    bids = [round(float(x), 4) for x in bids]
                    llm_bids = torch.tensor(bids, device=self.device, dtype=torch.float32)

                    if diff_items:
                        menu, menu_dict, payment_to_menu_option = Utilities.get_menu(
                            info["dist"], menu_cache, self.device)
                        
                    if bids == info["valuations"]:
                        truth_counter += 1
                    curr_regret, option = self.calculate_regret_batched(
                        menu, llm_bids, info["valuations_tensor"], payment_to_menu_option
                    )
                    regret += curr_regret

                    if verbose and (info["round"]%15 == 0 or info["round"] == 1):
                        Utilities.print_round_result(
                            info["round"], info["valuations"], bids, curr_regret, option, reasoning_text)
                return truth_counter, regret


            elif hist:
                for curr_round in range(1, self.rounds+1):
                    if diff_items:
                        vi_bar = random.choice([1, 1.5, 2, 2.5])
                        self.dist = [1,vi_bar]
                        menu, menu_dict, payment_to_menu_option = Utilities.get_menu(self.dist, menu_cache, self.device)
            
                    if not same_bidder:
                        valuations_tensor = llm_bidder.make_valuation(self.dist)
                        valuations = [round(float(x), 4) for x in valuations_tensor]

                    key = (self.dist[0], self.dist[1])
                    if key not in options_history_keys:
                        options_str = ", ".join(
                            f"{k}: Allocations = {allocs} & payment = {payment:.4f}"
                            for k, (allocs, payment) in menu_dict.items()
                        )
                        options_history.append(
                            f"Menu options for valuations domains [0,{self.dist[0]}] and [0,{self.dist[1]}]: "
                            f"{options_str}.\n")
                        options_history_keys.add(key)
                    history_strings = options_history + round_history

                    instruction = Prompts.game_instruction(auction_type, valuations, self.dist, if_history=hist,
                                                            history = history_strings, if_rule=rule,)

                    bid_text, reasoning_text = llm_bidder.make_bid(instruction)
                    bids = bid_text.strip("()").split(",")
                    bids = [round(float(x), 4) for x in bids]
                    llm_bids = torch.tensor(bids, device=self.device, dtype=torch.float32)

                    if bids == valuations:
                        truth_counter += 1 
                    curr_regret, option = self.calculate_regret_batched(menu, llm_bids, valuations_tensor, payment_to_menu_option)
                    regret += curr_regret
        
                    round_history.append(
                        f"Round {curr_round}: The previous bids, {bids[0]} for the apple and {bids[1]} for the banana, "
                        f"were made from these valuations domains:[0,{self.dist[0]}] and [0,{self.dist[1]}]."
                        f"This returned {option}.\n")

                    if truncate_history:
                        if history_limit is not None:
                            hist_limit = history_limit
                        elif reasoning == "medium":
                            hist_limit = 10
                        elif reasoning == "low":
                            hist_limit = 15
                        else:
                            hist_limit = 20

                        if hist_limit <= 0:
                            round_history.clear()
                        elif len(round_history) > hist_limit:
                            del round_history[:-hist_limit]

                    if verbose and (curr_round%15 == 0 or curr_round == 1):
                        Utilities.print_round_result(curr_round, valuations, bids, curr_regret, option, reasoning_text)
                return truth_counter, regret


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
    
    @staticmethod
    def get_menu(dist, menu_cache, device):
        key = (dist[0], dist[1])
        if key not in menu_cache:
            menu = Utilities.load_model(dist, device)
            X, p = menu.mech()
            menu_dict = Utilities.print_menu_options(X.detach(), p)
            payment_to_menu_option = {payment: k for k, (alloc, payment) in menu_dict.items()}
            menu_cache[key] = (menu, menu_dict, payment_to_menu_option)
        return menu_cache[key]
    
    @staticmethod
    def print_round_result(curr_round, valuations, bids, curr_regret, option, reasoning=""):
        print(f"ROUND {curr_round}:")
        print(f"  Valuations: {valuations} ")
        print(f"  Bids: {bids} ")
        if reasoning:
            print(f"  Reasoning: {reasoning}")
        if bids != valuations:
            print("  UNTRUTHFUL BID")
            print(f"  Regret this round: {curr_regret} ")
        print(f"  {option}")
    
