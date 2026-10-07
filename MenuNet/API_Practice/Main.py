import Objects
import Prompts

class Game:
    def __init__(self, args):
        self.args = args
        self.api_key = args["api key"]
        self.max_round = args["rounds"]
        self.N = args["number of agents"]
        self.persona_codes = args["persona code"]
        self.winning_guesses = []
        self.winner_history = []
        self.previous_rounds_history = []
        self.players = Objects.Player.generate_players(self.N, self.persona_codes, self.api_key)
        self.referee = Objects.Referee(self.api_key)

    def play_game(self):
        for round in range(1, self.max_round + 1):
            round_choices = []
            #criteria = self.format_input("Criteria: ")
            #albums = ["Undoing Ruin, Darkest Hour", "Shogun, Trivium", "Shadows are Security, AILD"]
            #criteria = albums[(round - 1)]
            criteria = self.referee.select_album("Criteria")
            game_instruction = Prompts.game_instruction("Standard", criteria)

            print(f"\n--- Round {round} ---")
            print(f"ALBUM: {criteria}\n")
            for player in self.players:
                guess = player.guess_with_reason(game_instruction)
                round_choices.append(guess)
                guess = player.format_response(guess)
                print(guess)

            #winner_ID = self.format_input("Enter the winner ID for this round: ")
            winner_ID = self.decide_winner(round_choices, criteria)
            self.previous_rounds_history.append(round_choices)
            self.winner_history.append(winner_ID)
            self.winning_guesses.append(
                next(item for item in round_choices if item["ID"] == int(winner_ID))["RECOMMENDATION"]
            )
            for player in self.players:
                if player.player_id == int(winner_ID):
                    player.score += 1
        
        self.show_history()
        self.final_winner()
        #self.referee.print_all_models()
                  



    def decide_winner(self, round_choices, criteria):
        round_winner = self.referee.decide_winner("Select", round_choices, criteria)
        print(f"Referee selected Player {round_winner} as the winner of this round.")
        return round_winner

    def final_winner(self):
        print("\n--- Final Scores ---")
        for player in self.players:
            print(f"Player {player.player_id}: {player.score}")
        winner = max(self.players, key=lambda p: p.score)
        winner_ids = [p.player_id for p in self.players if p.score == winner.score]
        if len(winner_ids) > 1:
            winners = ""
            for i in range(len(winner_ids)):
                if i == len(winner_ids)-1:
                    winners += str(winner_ids[i])
                else:
                    winners += (str(winner_ids[i])+", ")
            print(f"\nDRAW: Players {winners} with {winner.score} points each")
        else:
            print(f"\nWINNER: Player {winner_ids[0]} with {winner.score} points")

    def show_history(self):
        print("\n--- Game History ---")
        for i in range(self.max_round):
            print(f"Round {i+1} winner: Player {self.winner_history[i]}: {self.winning_guesses[i]}")

    def format_input(self, criteria = ""):
        user = input(criteria)
        if user in ["quit", "clear", "q"]:
            print("\nEXIT")
            quit()
        return user

            




if __name__ == "__main__":
    args = {
        "number of agents": 3,
        "rounds": 1,
        "api key": "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",
        "persona code": ["1","2","3"],
    }
    game = Game(args)
    game.play_game()