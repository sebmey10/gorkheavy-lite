import requests
import json

# These are the sockets of each container that I'm going to deploy.
api_endpoints = {
    "promptimizer_granite": "http://promptimizer-service:11434/api/generate",
    "llama": "http://llama-service:11434/api/generate",
    "qwen": "http://qwen-service:11434/api/generate",
    "qwen_small": "http://qwen-small-service:11434/api/generate",
    "judge": "http://judge-service:11434/api/generate",
}

# These are the models I'm using to execute the workflow
models = {
    "promptimizer": "granite4:350m",
    "llama": "llama3.2:1b-instruct-q4_0",
    "qwen": "qwen2.5-coder:1.5b-instruct-q4_0",
    "qwen_small": "qwen3:0.6b",
    "judge": "gemma3:1b"
}

def promptimizer(user_input):
    promptimizer_prompt = f"""
    Take {user_input} and rewrite it into a more concise query. The goal is to provide AI systems with a clear, 
    focused prompt for optimal interpretation and response. Only respond with the re-written query."""

    json_promptimizer = {
        "model": models["promptimizer"],
        "prompt": promptimizer_prompt,
        "stream": False
    }

    try:
        send_promptimizer = requests.post(api_endpoints["promptimizer_granite"], json= json_promptimizer)
        send_promptimizer.raise_for_status()
        response_promptimizer = send_promptimizer.json()
        message_promptimizer = response_promptimizer["response"]

    except requests.exceptions.RequestException as failed:
        raise Exception(f"Didn't work: {str(failed)}")

    return message_promptimizer


llama_logfile = []
qwen_logfile = []
qwen_small_logfile = []


def send_message_models(user_input):

    optimized_prompt = promptimizer(user_input)

    json_qwen_small = {
        "model": models["qwen_small"],
        "prompt": optimized_prompt,
        "stream": False
    }
    
    json_llama = {
        "model": models["llama"],
        "prompt": optimized_prompt,
        "stream": False
    }

    json_qwen = {
        "model": models["qwen"],
        "prompt": optimized_prompt,
        "stream": False
    }


    try:
        send_qwen_small = requests.post(api_endpoints["qwen_small"], json=json_qwen_small)
        send_qwen_small.raise_for_status()
        response_qwen_small = send_qwen_small.json()
        message_qwen_small = response_qwen_small["response"]
        qwen_small_logfile.append({"role": "assistant", "content": message_qwen_small})

        send_llama = requests.post(api_endpoints["llama"], json=json_llama)
        send_llama.raise_for_status()
        response_llama = send_llama.json()
        message_llama = response_llama["response"]
        llama_logfile.append({"role": "assistant", "content": message_llama})

        send_qwen = requests.post(api_endpoints["qwen"], json=json_qwen)
        send_qwen.raise_for_status()
        response_qwen = send_qwen.json()
        message_qwen = response_qwen["response"]
        qwen_logfile.append({"role": "assistant", "content": message_qwen})


        return message_qwen_small, message_llama, message_qwen
    
    except requests.exceptions.RequestException as failed:
        raise Exception(f"Didn't work: {str(failed)}")
    

def make_judgement(user_input):

    judge_prompt = f"""
    User query: {user_input}

    qwen_small answer: {qwen_small_logfile[-1]['content']}
    LLaMA answer: {llama_logfile[-1]['content']}
    Qwen answer: {qwen_logfile[-1]['content']}

    Choose the best answer based on correctness, completeness, clarity, and usefulness.
    Return the contents of the best answer, nothing else.
    """

    
    json_judge = {
        "model": models["judge"],
        "prompt": judge_prompt,
        "stream": False
    }

    try:
        send_judge = requests.post(api_endpoints["judge"], json=json_judge)
        send_judge.raise_for_status()
        response = send_judge.json()
        message_judge = response["response"]
        return str(message_judge)
    
    except requests.exceptions.RequestException as failed:
        raise Exception(f"Didn't work: {str(failed)}")

print("Chatting with gorkheavy-lite! (type 'exit' to quit)")

while True:

    user_input = input("YOU: ").strip()

    if user_input.lower() == "exit":
        print("Bye!")
        break

    try:
        send_message_models(user_input)
        reply = make_judgement(user_input)
        print(f"Reply: {reply}\n")
    except Exception as failed:
        print(f"Error: {failed}")



