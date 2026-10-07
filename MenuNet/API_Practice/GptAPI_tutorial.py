from openai import OpenAI

client = OpenAI(api_key="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")

messages = []

def run():
    while True:

        user_input = input("\nYou: ")
        if user_input in ["quit", "clear", "q"]:
            print("\nEXIT")
            break
        if user_input == "messages":
            print("\nMessages:")
            for msg in messages:
                print(f"{msg['role']}: {msg['content']}")
            continue

        messages.append({"role": "USER", "content": user_input})
                        
        response = client.responses.create(
            model = "gpt-5-mini",
            instructions = "you are a metal expert",
            input = user_input
        )

        messages.append({"role": "ASSISTANT", "content": response.output_text})
        print(response.output_text)

if __name__ == "__main__":
    run()
