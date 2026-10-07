import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats
import os
import ast
    
class data_processor():
    def __init__(self, csv_name):
         self.csv_name = csv_name
         self.csv_path = "MenuNet/Auction_Results/" + self.csv_name

    def store_data(self, summary):
        folder_path = os.path.join(os.path.dirname(__file__), "Auction_Results")
        os.makedirs(folder_path, exist_ok=True)
        df = pd.DataFrame([summary])
        file_exists = os.path.isfile(self.csv_path)
        df.to_csv(self.csv_path, mode="a", header = not file_exists, index = False)


    def t_test(self, category, index):
        df = pd.read_csv(self.csv_path)
        df1 = df.loc[df[category], ["avg_truth", "avg_regret"]]
        df2 = df.loc[df[category]==False, ["avg_truth", "avg_regret"]]
        t_stat, p_val = stats.ttest_ind(df1[index], df2[index])
        print(f"t-test for {index} comparing with and without {category}")
        print(f"  t-statistic: {t_stat:.4f}")
        print(f"  p-value: {p_val:.4f}")
        if p_val <= .05:  # type: ignore
            print("Difference is Significant")


    def menu_option_means(self, category):
        df = pd.read_csv(self.csv_path)
        df1 = df.loc[df[category], ["aggregate_menu_counts"]]
        df2 = df.loc[df[category]==False,["aggregate_menu_counts"]]
        df1_expanded = pd.DataFrame([ast.literal_eval(x) for x in df1["aggregate_menu_counts"]])
        df2_expanded = pd.DataFrame([ast.literal_eval(x) for x in df2["aggregate_menu_counts"]])
        print(df1_expanded.mean(), "\n\n", df2_expanded.mean(), "\n")
        for option in df1_expanded.columns:
            t_stat, p_val = stats.ttest_ind(df1_expanded[option], df2_expanded[option])
            print(f"{option}:")
            print(f"  t-statistic: {t_stat:.4f}")
            print(f"  p-value: {p_val:.4f}")


    def create_plots(self, category):
        df = pd.read_csv(self.csv_path)
        df1 = df.loc[df[category], ["avg_truth", "avg_regret"]]
        df2 = df.loc[df[category]==False, ["avg_truth", "avg_regret"]]
        print(f"{category}: \n{df1.mean()}")
        print(f"No {category}: \n{df2.mean()}")
        
        fig, axes = plt.subplots(1, 2, figsize=(10, 5))

        axes[0].boxplot([df2["avg_truth"], df1["avg_truth"]], 
                        tick_labels=["Without History", "With History"])
        axes[0].set_title('Average Truth: With History vs Without History')
        axes[0].set_ylabel('Average Truth')

        axes[1].boxplot([df2["avg_regret"], df1["avg_regret"]], 
                        tick_labels=["Without History", "With History"])
        axes[1].set_title('Average Regret: With History vs Without History')
        axes[1].set_ylabel('Average Regret')

        plt.tight_layout()
        plt.show()
