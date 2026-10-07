from openai import OpenAI
import torch


class Agent:
    def __init__(self, api_key="", model='gpt-5-mini'):
        self.model = model
        #MODELS: gpt-5-mini, gpt-5.2, gpt-4o
        self.client = OpenAI(api_key=api_key)

    def communicate(self, context, instructions=""):
        prompt = context + "\n"
        message = ""

        response = self.client.responses.create(
            model=self.model,
            #instructions=instructions,
            #temperature = self.temp,
            input=prompt,
            
            #These are NOT supported by 4o
            text = {"verbosity": "low"},
            reasoning = {"effort": "minimal"},
            service_tier="flex"
        )

        message = response.output_text
        #print(message)
        return message
    

class LLMBidder(Agent):
    def __init__(self, api_key="", model='gpt-5-mini'):
        super().__init__(api_key, model)
        self.valuations = None
        self.bid = None

    def make_valuation(self, max_list):
        item1_max, item2_max = max_list
        valuation1 = item1_max * torch.rand((1,1,1)) 
        valuation2 = item2_max * torch.rand((1,1,1))
        self.valuations = torch.cat((valuation1, valuation2), dim=2).flatten()
        return self.valuations

    def make_bid(self, context=""):
        self.bid = self.communicate(context)
        return self.bid
        
    



    