Persona_Details = {
    "1": "a player who only knows prefers 1990s and 2000s MeloDeath.",
    "2": "a player who only knows 2000s Metalcore.",
    "3": "a player who only knows Japanese Metal."
}

Game_Rule = {
    "Standard": "Welcome to the game. \
In this game, each player will recommend 1 album similar to the the album provided. After all players \
have made their recommendations, the prompter will select a winner and that player will earn 1 score. \
The highest score at the end of all rounds wins the game. You MUST format your answer as: \
REASONING: (up to 3 sentences) and RECOMMENDATION: album (year), artist. \n",

    "Test": "\nProvide me 3 albums similar to the one provided: "
}

Referee_Prompts = {
    "Criteria": "Select any metal album from 1997 to 2009. Respond ONLY in this format: Album (year), Artist.",

    "Select": "You are the referee of this game. Your role is to evaluate the recommendations provided \
by each player based on how closely they align with the given album. After all players have submitted \
their recommendations, you will determine which player's recommendation is the best match to the \
album and declare them the winner of the round. Respond with ONLY the integer of the winner's ID.\n"
}

def game_instruction(rule_type, criteria):
    instruction = ""
    instruction += Game_Rule[rule_type]
    instruction += f"ALBUM: {criteria}"
    return instruction
