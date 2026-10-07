import Auction
from Data_Process import data_processor
from datetime import datetime

class Auction_Runner():
    def __init__(self):
        self.distribution = [1,2]
        #self.csv_name = f"[0,{self.distribution[0]}] x [0,{self.distribution[1]}].csv"
        self.csv_name =  F"History Comp.csv"
        self.data = data_processor(self.csv_name)

    def Run_Auction(self, rule=False, hist=True, same_bidder=True, 
                    diff_items = False, model='gpt-5-mini'):
        runs = 10
        rounds = 30
        distribution = self.distribution
        auction = Auction.Auction(rounds, distribution)
        truths_per_run = []
        regret_per_run = []
        aggregate_menu_counts = {}

        title = f"w{'/' if hist else '/o'} History"
        title += " w/ Different Items Valuations" if diff_items else ""
        title += " w/ Rule" if rule else ""
        title += " w/ Same Bidder" if same_bidder else " w/ Different Bidders"

        method = auction.auction
        
        for curr_run in range(1, runs+1):
            print(f"\n---RUN {curr_run}---")
            truth_counter, regret, menu_option_counter = method(hist=hist, 
            rule= rule, same_bidder=same_bidder, diff_items=diff_items, model=model)
            
            truths_per_run.append(truth_counter)
            regret_per_run.append(regret)
            for key, count in menu_option_counter.items():
                aggregate_menu_counts[key] = aggregate_menu_counts.get(key, 0) + count
            
        avg_truth = sum(truths_per_run)/curr_run
        avg_regret = sum(regret_per_run)/curr_run
        print(f"\nFINAL TOTALS {title}")
        print("  Truth AVG:", avg_truth)
        print("  Truth per run:", truths_per_run)
        print("  Regret AVG:", avg_regret)
        print("  Regret per run:", regret_per_run)
        if aggregate_menu_counts:
            print("Aggregate menu option counts over all runs:")
            for key, count in aggregate_menu_counts.items():
                print(f"  {key}: {count}")
        
        summary = {
            "date": datetime.now().strftime("%m/%d/%Y, %H:%M"),
            "rule": rule,
            "hist": hist,
            "same_bidder": same_bidder,
            "diff_items": diff_items,
            "model": model,
            "runs": runs,
            "rounds": rounds,
            "avg_truth": avg_truth,
            "avg_regret": avg_regret,
            #"truths_per_run": str(truths_per_run),
            #"regret_per_run": regret_per_run,
            #"aggregate_menu_counts": str(aggregate_menu_counts),
        }
        #self.data.store_data(summary)
    
    def analysis(self):
        #self.data.t_test("hist", "avg_truth")
        #self.data.t_test("hist", "avg_regret")
        self.data.create_plots("hist")
        #self.data.menu_option_means("hist")

    def run_mulitple(self):
        for i in range(1):
            #no rule, diff bidders
            #run.Run_Auction(rule=False, hist=True, same_bidder=False)
            #run.Run_Auction(rule=False, hist=False, same_bidder=False)
            #no rule, same bidders
            run.Run_Auction(rule=False, hist=True, same_bidder=True)
            #run.Run_Auction(rule=False, hist=False, same_bidder=True)
            #rule, diff bidders
            #run.Run_Auction(rule=True, hist=True, same_bidder=False)
            #run.Run_Auction(rule=True, hist=False, same_bidder=False)
            #rule, same bidders
            run.Run_Auction(rule=True, hist=True, same_bidder=True)
            #run.Run_Auction(rule=True, hist=False, same_bidder=True)
            
if __name__ == "__main__":
    run = Auction_Runner()
    run.run_mulitple()
    #run.Run_Auction(rule=True, hist=True, same_bidder=True)
    #run.analysis()

#   python MenuNet\Auction_Runner.py