# import requests
# import json

# ollama_url = "http://127.0.0.1:11434"
# model_name = "gemma3-1b"

# def chat_with_model(prompt):

#     payload = {
        
#         "model": model_name,
#         "prompt": prompt,
#         "stream": True
#     }

#     try:
#         response = requests.post(ollama_url, json = payload)

#         if response.status_code == 200:

#             result = response.json()

#             return result['response']
        
#         else:
#             return f"Error: Received status code {response.status_code}"
        
#     except requests.exceptions.RequestException as e:
#         return f"Error: {str(e)}"
    
# def main():
#     print("Chat with Gemma")
#     print("Type 'quit' or 'exit' to end the chat.")
#     print("-" * 50)

#     while True:
#         user_input = input("n\You: ")

#         if user_input.lower() in ['quit','exit']:
#             print("Goodbye")
#             break
#         print("\nGemma: ", end="", flush=True)
#         response = chat_with_model(user_input)
#         print(response)

# if __name__ == "__main__":
#     main()


from ollama import chat
from ollama import ChatResponse
import json
import requests

model = "gemma3:1b"

logfile = []

user_input = input("YOU: ")

input = user_input.strip()

model_output = (f"Gemma: ")

ollama_url = "http://127.0.0.1:11434"

def chat(user_input):

    response: ChatResponse = chat(model=model, messages = [
        {
            'role': 'user',
            'content': user_input
        }
    ])

    logfile.append(response.message.content)

    payload = {
        "model": model,
        "prompt": user_input,
        "stream": True
    }

    response = requests.posts(ollama_url, data = response)
    
    return response

def main():

    while True:
        input = input

        if input.lower() in ["quit"]:
            print("Goodbye")
            break
        print("Gemma: ", end = "")
        response = response
        print(response)
    return response
    