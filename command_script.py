import requests
import json

# These are the sockets of each container that I'm going to deploy.
api_endpoints = {
    "gemma": "http://127.0.0.1:11434/api/generate",
    "llama": "http://127.0.0.1:11435/api/generate",
    "qwen": "http://127.0.0.1:11436/api/generate",
    "judge": "http://127.0.0.1:11437/api/generate"
}

# These are the models I'm using to execute the workflow
models = {
    "gemma": "gemma3:1b",
    "llama": "llama3.2:1b-instruct-q4_0",
    "qwen": "qwen2.5-coder:1.5b-instruct-q4_0",
    "judge": "gemma3:4b"
}



gemma_logfile = []
llama_logfile = []
qwen_logfile = []


def send_message_models(user_input):

    json_gemma = {
        "model": models["gemma"],
        "prompt": user_input,
        "stream": False
    }
    
    json_llama = {
        "model": models["llama"],
        "prompt": user_input,
        "stream": False
    }

    json_qwen = {
        "model": models["qwen"],
        "prompt": user_input,
        "stream": False
    }


    try:
        send_gemma = requests.post(api_endpoints["gemma"], json=json_gemma)
        send_gemma.raise_for_status()
        response_gemma = send_gemma.json()
        message_gemma = response_gemma["response"]
        gemma_logfile.append({"role": "assistant", "content": message_gemma})

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


        return gemma_logfile, llama_logfile, qwen_logfile
    
    except requests.exceptions.RequestException as failed:
        raise Exception(f"Didn't work: {str(failed)}")
    

def make_judgement(user_input):

    judge_prompt = f"""
    User query: {user_input}

    Gemma answer: {gemma_logfile[-1]['content']}
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
    
while True:
    print("Chatting with gorkheavy-lite! (type 'exit' to quit)")
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



