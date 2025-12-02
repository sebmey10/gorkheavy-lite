import asyncio
import aiohttp
import sys
import os

# These are the sockets of each container that I'm going to deploy.
api_endpoints = {
    "promptimizer": "http://promptimizer:11434/api/generate",
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


async def promptimizer(session, user_input):
    prompt_text = f"""
You are an expert Prompt Engineer and Logic Optimizer. Your goal is to rewrite {user_input}\n
to be precise, concise, and highly actionable for an AI model.

Follow these steps for every input:
1. Identify the Core Intent: What is the user actually trying to achieve?
2. Remove Fluff: Delete polite filler (e.g., "Please," "I was wondering"), vague descriptors, and redundant context.
3. Clarify Constraints: Explicitly state the desired format, length, or style if implied.
4. Structure: Use bullet points or step-by-step instructions if the task is complex.

Output Format:
Provide ONLY the optimized prompt in maximum 4 sentences. Do not add conversational filler.
"""

    json_promptimizer = {
        "model": models["promptimizer"],
        "prompt": prompt_text,
        "stream": False
    }

    try:
        async with session.post(api_endpoints["promptimizer"], json=json_promptimizer, timeout=aiohttp.ClientTimeout(total=120)) as response:
            response.raise_for_status()
            data = await response.json()
            message = data["response"]
            return message
    except aiohttp.ClientError as f:
        print(f"Promptimizer failed: {f}")
        return user_input
    

async def send_qwen_small(session, prompt):

    json_qwen_small = {
        "model": models["qwen_small"],
        "prompt": prompt,
        "stream": False
        }

    try:
        async with session.post(api_endpoints["qwen_small"], json = json_qwen_small, timeout=aiohttp.ClientTimeout(total=120)) as qs:
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
        async with session.post(api_endpoints["qwen"], json = json_qwen, timeout=aiohttp.ClientTimeout(total=120)) as q:
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
        async with session.post(api_endpoints["llama"], json = json_llama, timeout=aiohttp.ClientTimeout(total=120)) as ll:
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
        send_qwen(session, optimized_prompt),
        send_llama(session, optimized_prompt),
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
    """

    json_judge = {
        "model": models["judge"],
        "prompt": judge_prompt,
        "stream": False
    }

    try:
        async with session.post(api_endpoints["judge"], json = json_judge, timeout=aiohttp.ClientTimeout(total=120)) as jud:
            jud.raise_for_status()
            data = await jud.json()
            response = data["response"]
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
