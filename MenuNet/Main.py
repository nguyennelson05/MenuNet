import Auction
from Data_Process import data_processor
from datetime import datetime

class Auction_Runner():
    def __init__(self):
        self.distribution = [1,2]
        #self.csv_name = f"old_data.csv"
        self.csv_name =  f"GPT-5.2_Medium.csv"
        self.data = data_processor(self.csv_name)

    def Run_Auction(self, rule=False, hist=True, same_bidder=True, 
                    diff_items = False, model='gpt-5.2', reasoning="medium"):
        runs = 1
        rounds = 30
        distribution = self.distribution
        auction = Auction.Auction(rounds, distribution)
        truths_per_run = []
        regret_per_run = []

        title = f"{'w/' if hist else 'w/o'} History"
        title += " w/ Different Items Valuations" if diff_items else ""
        title += " w/ Rule" if rule else ""
        title += " w/ Same Bidder" if same_bidder else " w/ Different Bidders"
        title += f"; {model}"
        title += f" w/ {reasoning} reasoning effort"
        method = auction.auction
        
        for curr_run in range(1, runs+1):
            print(f"\n---RUN {curr_run}---")
            truth_counter, regret = method(rule, hist, same_bidder, diff_items, model, reasoning)
            
            truths_per_run.append(truth_counter)
            regret_per_run.append(regret)
            
        avg_truth = sum(truths_per_run)/curr_run
        avg_regret = sum(regret_per_run)/curr_run
        print(f"\nFINAL TOTALS {title}")
        print("  Truth AVG:", avg_truth)
        print("  Truth per run:", truths_per_run)
        print("  Regret AVG:", avg_regret)
        print("  Regret per run:", regret_per_run)
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
        }
        #IMPORTANT-----------------
        self.data.store_data(summary)
        #IMPORTANT-----------------
    
    
    def analysis(self, params = []):
        #self.data.menu_option_means("hist")
        self.data.compare_auctions(params[0], params[1], params[2])

    def run_mulitple(self):
        for i in range(30):
            #no rule, diff bidders
            run.Run_Auction(rule=False, hist=True, same_bidder=False)
            run.Run_Auction(rule=False, hist=False, same_bidder=False)
            #no rule, same bidders
            run.Run_Auction(rule=False, hist=True, same_bidder=True)
            run.Run_Auction(rule=False, hist=False, same_bidder=True)
            #rule, diff bidders
            run.Run_Auction(rule=True, hist=True, same_bidder=False)
            run.Run_Auction(rule=True, hist=False, same_bidder=False)
            #rule, same bidders
            run.Run_Auction(rule=True, hist=True, same_bidder=True)
            run.Run_Auction(rule=True, hist=False, same_bidder=True)
            #diff items does not work with same_bidder, be careful
            
if __name__ == "__main__":
    run = Auction_Runner()
    #run.run_mulitple()
    #run.Run_Auction(rule=True, hist=True, same_bidder=True, diff_items=False, reasoning="minimal", model='gpt-5-mini')
    run.analysis([1,1,1])

#   python MenuNet\Auction_Runner.py
