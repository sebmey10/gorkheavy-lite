import requests 
import json

ollama_url = "http://127.0.0.1:11434/api/generate"

model = "gemma3:1b"

logfile = []

def send_message(user_input):

    json_payload = {
        "model": model,
        "messages": [{"role": "user", "content": user_input}],
        "stream": False
    }

    send_input = requests.posts(ollama_url, data = json_payload)

    send_input.raise_for_status()

    response = send_input.json()
    print(response)

    message = response["message"]["content"]
    print(message)

    logfile.append(message)

    return message

def main():
    print("Chatting with Ollama(type 'quit' to exit)")

    while True:

        user_input = input("YOU: ").strip()

        if user_input.lower() in ["quit"]:
            print("Goodbye")
            break

        try:
            reply = send_message(user_input)
            print(f"Gemma: {reply}\n")

        except Exception as e:
            print(f"Error {e}")
