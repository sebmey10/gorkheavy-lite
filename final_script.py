import asyncio
import aiohttp
import json
"""gc.collect in order to clear unwanted memory when I want to."""

# These are the sockets of each container that I'm going to deploy.
api_endpoints = {
    "promptimizer_granite": "http://promptimizer:11434/api/generate",
    "llama": "http://llama:11434/api/generate",
    "qwen": "http://qwen:11434/api/generate",
    "qwen_small": "http://qwen-small:11434/api/generate",
    "judge": "http://judge:11434/api/generate",
}

# These are the models I'm using to execute the workflow
models = {
    "promptimizer": "granite4:350m",
    "llama": "llama3.2:1b",
    "qwen": "qwen2.5-coder:1.5b",
    "qwen_small": "qwen3:0.6b",
    "judge": "gemma3:1b"
}

# Note: We're using lists instead of appending to maintain conversation history

conversation_memory = []


async def promptimizer(session, user_input):
    promptimizer = f"""
    Take {user_input} and rewrite it into a more concise query. The goal is to provide AI systems with a clear, 
    focused prompt for optimal interpretation and response. Only respond with the re-written query.
    """

    json_promptimizer = {
        "model": models["promptimizer"],
        "prompt": promptimizer,
        "stream": False
    }

    try:
        async with session.post(api_endpoints["promptimizer_granite"],json=json_promptimizer) as response:
            response.raise_for_status()
            data = await response.json()
            message = data["response"]
            conversation_memory.append({"role": "promptimizer", "content": message})

            return message
    except aiohttp.ClientError as f:
        print(f"Promptimizer failed: {f}")
    

async def send_qwen_small(session, prompt):

    json_qwen_small = {
        "model": models["qwen_small"],
        "prompt": prompt,
        "stream": False
        }

    try:
        async with session.post(api_endpoints["qwen_small"], json = json_qwen_small) as qs:
            qs.raise_for_status()
            data = await qs.json()
            message = data["response"]
            return message
    except aiohttp.ClientError as e:
        raise Exception(f"Failed at qwen_small: {e}")
    

async def send_qwen(session, prompt):
    
    json_qwen = {
        "model": models["qwen"],
        "prompt": prompt,
        "stream": False
    }

    try:
        async with session.post(api_endpoints["qwen"], json = json_qwen) as q:
            q.raise_for_status()
            data = await q.json()
            response = data["response"]
            return response
        
    except aiohttp.ClientError as e:
        raise Exception(f"Failed at qwen: {e}")
    

async def send_llama(session, prompt):

    json_llama = {
        "model": models["llama"],
        "prompt": prompt,
        "stream": False
    }

    try:
        async with session.post(api_endpoints["llama"], json = json_llama) as ll:
            ll.raise_for_status()
            data = await ll.json()
            response = data["response"]
            return response
    
    except aiohttp.ClientError as e:
        raise Exception(f"Failed at llama: {e}")
    

async def send_all_models(session, user_input):

    optimized_prompt = await promptimizer(session, user_input)

    send = await asyncio.gather(
        send_qwen_small(session, optimized_prompt),
        send_qwen(session, user_input),
        send_llama(session, user_input),
        return_exceptions = True
    )

    for s in send:
        if isinstance(s, Exception):
            raise s
    
    return send[0], send[1], send[2]


async def send_judge(session, user_input, qwen_small_answer, llama_answer, qwen_answer):
    """Have the judge model select the best answer."""
    judge_prompt = f"""
    User query: {user_input}

    qwen_small answer: {qwen_small_answer}
    LLaMA answer: {llama_answer}
    Qwen answer: {qwen_answer}

    Choose the best answer based on correctness, completeness, clarity, and usefulness.
    Return the contents of the best answer, nothing else.

    To reference context of the conversation you're having, reference {conversation_memory}.
    """

    json_judge = {
        "model": models["judge"],
        "prompt": judge_prompt,
        "stream": False
    }


    try:
        async with session.post(api_endpoints["judge"], json = json_judge) as jud:
            jud.raise_for_status()
            data = await jud.json()
            response = data["response"]
            conversation_memory.append({"role":"judge", "content": response})
            return str(response)
        
    except aiohttp.ClientError as e:
        raise Exception(f"Failed at judge: {e}")
    

async def main():

    print("You now have the pleasure of speaking with Gork,\n" \
    "the world's closest attempt to AGI.\n" \
    "Type 'exit' to quit.")

    async with aiohttp.ClientSession() as session:
        while True:

            user_input = await asyncio.get_event_loop().run_in_executor(
                None, lambda: input("YOU: ")
            )

            if user_input.lower() == "exit":
                break

            try:
                qwen_small_response, qwen_response, llama_response = await send_all_models(session, user_input)
                reply = await send_judge(session, user_input, qwen_small_response, llama_response, qwen_response)

                print(f"Reply: {str(reply)}")

            except Exception as failed:
                print(f"Error {failed}")

                


if __name__ == "__main__":
    asyncio.run(main())
