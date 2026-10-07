from openai import OpenAI
import Prompts as prompts

class Agent:
    def __init__(self, api_key = "", temperature=1, model="gpt-5.2"):
        self.temp = temperature
        self.model = model
        self.client = OpenAI(api_key=api_key)
        self.resp = self.client.models.list()

    def communicate(self, context, instructions=""):
        prompt = context + "\n"
        message = ""

        response = self.client.responses.create(
            model=self.model,
            instructions=instructions,
            input=prompt,
            temperature=self.temp,
            #service_tier="flex"
        )

        message = response.output_text
        #print(message)
        return message
    
    def print_all_models(self):
        print("\nMODELS:")
        for m in self.resp.data:
            print(m.id)
    




class Player(Agent):
    def __init__(self, player_id, persona_code, api_key="", temperature=1, model='gpt-4o'):
        Agent.__init__(self, api_key, temperature, model)
        self.player_id = player_id
        self.persona_code = prompts.Persona_Details[persona_code]
        self.score = 0
        self.history = []

    def guess_with_reason(self, context):
        if self.history:
            context = context + f" Your previous recommendations were: {', '.join(self.history)}"
        #print(context)
        guess = self.communicate(context, self.persona_code).split("\n")
        guess_dict = {}
        if "" in guess: guess.remove("")
        for item in guess:
            if item.startswith('REASONING:'):
                guess_dict['REASONING'] = item.replace('REASONING:', '').strip()
            elif item.startswith('RECOMMENDATION:'):
                guess_dict['RECOMMENDATION'] = item.replace('RECOMMENDATION:', '').strip()
        guess_dict['ID'] = self.player_id
        self.history.append(guess_dict['RECOMMENDATION'])
        return guess_dict
    
    def format_response(self, guess):
        response = f"Player {guess["ID"]} Guess: {guess["RECOMMENDATION"]}\n{guess["REASONING"]}\n"
        return response

    @staticmethod
    def generate_players(agents, persona_codes, api_key=""):
        players = []
        for i in range(agents):
            players.append(Player(player_id=i+1, persona_code=persona_codes[i], api_key=api_key))
        return players
    



class Referee(Agent):
    def __init__(self, api_key="", temperature=1, model='gpt-4o'):
        Agent.__init__(self, api_key, temperature, model)
        self.selection_history =[]

    def select_album(self, context):
        instruction = prompts.Referee_Prompts[context]
        if self.selection_history:
            instruction = instruction + f" Previously selected albums: {', '.join(self.selection_history)}" 
        #print(instruction)
        album = self.communicate(instruction)
        self.selection_history.append(album)
        return album
    
    def decide_winner(self, context, round_choices, criteria):
        instruction = prompts.Referee_Prompts[context]
        context = f"ALBUM: {criteria} {instruction}"
        for guess in round_choices: 
            guess = f"Player {guess['ID']} Recommendation: {guess['RECOMMENDATION']}\nReasoning: {guess['REASONING']}\n"
            context += guess
        #print(context)
        winner = self.communicate(context, instruction)
        return winner