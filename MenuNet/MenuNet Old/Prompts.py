Auction_Type = {
    "Auction": """
Imagine you are participating in a First-price auction for an apple item.
All bidders submit sealed bids.
The highest bidder wins the auction and pays the amount they bid.
If multiple people bid the highest price at the same time, one winner will be randomly selected.
The true value of the item to you is not disclosed.
Please provide a strategic bid amount that maximizes your chance of winning the auction
while minimizing the cost. Remember, your goal is to outbid others without overpaying.
Return only your bid as an integer value.
""",

    "MenuNet": """
This is an auction hosted by a neural network. You are the only bidder and there are 2 items; an apple and an banana. 
You will place a bid for the items based on your valuations.
Your bids will be given to the neural network model, and the model will then return a menu option, consisting of an allocation and payment.
Allocations are in the form of [alloc1, alloc2], with the first value being the allocation to the apple, and the second value being the allocation for the banana.
Your expected utility is (your value of the items) * allocation (the probability that the model assigns the items to you) - 
payment (ammount you need to pay that the model determines based on all bids for the items; the payment is a float.).
YOUR TOP PRIORITY is to place bids that MAXIMIZE your actual UTILITY in the long run.
Both the probability that the model assigns the items to you and the payment you need to pay are non-negative.
When the model calculates the utility, it will regards your actual bids as your values of the items and ensures that the utility it calculates is non-negative.
So, only when your actual bids are not exceeding your values of the items, the model can fully ensure that the actual utility is non-negative.
The apple is always the first item, the banana is always the second item.
Your value of the apple is {valuations[0]} selected from domain [0,{valuation_domain[0]}]. 
Your value of the banana is {valuations[1]} selected from domain [0,{valuation_domain[1]}].
{rule}
Give your response with a tuple, arranging your bids for each item in the order which the items are valued, without using other text, 
e.g. (0.2254, 0.4187,...). The bid values are limited to four decimal places maximum. 
If bidding truthfully, you bid the EXACT SAME as the valuation, to the same decimal place.
Example, if your valuation equals .979, your bid should equal .979, NOT .98 or .9789.
Don't reverse the bidding order of items. Follow the order in which the items are valued.
You only need to state the tuple of your bids, no further explanation or additional text is required. JUST A TUPLE IS NEEDED.
Your bids can be untruthful. Your bids can be different from your valuation as long as you optimize your utility. 
""",
}

def game_instruction(auction_type, valuations:list, valuation_domain:list, if_history:bool, history:list, if_rule:bool):
    instruction = ""
    rule=""
    if auction_type == "MenuNet":
        if if_rule:
            rule += f"""
The model is a neural network architecture designed to optimize auction mechanisms.
It has been trained specifically for the domain ([0,{valuation_domain[0]}] and [0,{valuation_domain[1]}]) your item valuations were selected from.
The architecture is comprised of 2 networks. The first network takes no input and outputs an allocation matrix and payment vector,
representing the possible menu options. The second network take the allocation matrix and payment vector as input,
and outputs a the buyer's strategy over all possible valuation profiles. This is later used to calculate the loss function, 
of which the architecture uses to maximize the auction mechanism's revenue.
"""
        instruction += Auction_Type["MenuNet"].format(valuations=valuations, rule=rule, valuation_domain=valuation_domain)
    if if_history == True:
        instruction += "\nThis is your previous round history:\n"
        for item in history:
            instruction += item
    #print(instruction)
    return instruction


#comparison: 1) no memory vs w/ memory 
#            2) items change per round (different item valuations) 
#            3) Bidder changes per round (different bidder valuations)
#            4) rule vs no rule 
#            5) different gpt models (5 mini, 5.2, 4o)

#default: same bidder, same items, w/ memory, no rule, gpt5 mini

#Motivation/Background: How effective are LLMs as bidders, how optimally they perform
#using truthfulness as a metric: truthuful > optimal option (IC)
#industry trends, EX: Alibaba used Neural Auctions to perform aucitons on their online advertisements
#EX: some companies use LLM bidders

#explain neural auctions, trend of more llm bidders
#figure, brief explanation of research: evaluating how effective llms are at neural auctions
#re use figure 2 but simplify, what is input and what is output

#Poster Content: Background & Purpose, Hypothesis Results, Compare FP, SP, & MenuNet
#Abstract 250-500 words, 1 page max, background, motivation, takeaways