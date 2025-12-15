import asyncio
import aiohttp
import sys
import logging
import csv
import os
import hashlib

# Set up basic logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    stream=sys.stdout
)
logger = logging.getLogger(__name__)

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
    logger.info("Starting promptimizer")
    
    prompt_text = f"""You are an expert prompt engineer. Transform the following user request into an optimal prompt that will produce the best possible response from an AI model.

USER REQUEST: {user_input}

OPTIMIZATION RULES:
1. Preserve the original intent completely
2. Add specificity: define scope, format, and constraints
3. Request step-by-step reasoning for complex tasks
4. Specify the desired output format (e.g., bullet points, code, explanation)
5. Remove ambiguity while keeping the prompt concise

OUTPUT: Return ONLY the optimized prompt with no preamble, explanation, or meta-commentary."""

    json_promptimizer = {
        "model": models["promptimizer"],
        "prompt": prompt_text,
        "stream": False
    }

    try:
        logger.info("Hit promptimizer API")
        async with session.post(api_endpoints["promptimizer"], json=json_promptimizer) as response:
            response.raise_for_status()
            data = await response.json()
            message = data["response"]
            logger.info("Promptimizer is good")
            return message
    except aiohttp.ClientError as f:
        logger.error(f"Promptimizer failed: {f}")
        return user_input
    

async def send_qwen_small(session, prompt):
    logger.info("Start qwen_small")

    json_qwen_small = {
        "model": models["qwen_small"],
        "prompt": prompt,
        "stream": False
        }

    try:
        logger.info("Hit qwen_small API")
        async with session.post(api_endpoints["qwen_small"], json=json_qwen_small) as qs:
            qs.raise_for_status()
            data = await qs.json()
            message = data["response"]
            logger.info("qwen_small is good")
            return message
    except aiohttp.ClientError as e:
        logger.error(f"qwen_small failed: {e}")
        raise Exception(f"Failed at qwen_small: {e}")
    

async def send_qwen(session, prompt):
    logger.info("Starting qwen")
    
    json_qwen = {
        "model": models["qwen"],
        "prompt": prompt,
        "stream": False
    }

    try:
        logger.info("Hit qwen API...")
        async with session.post(api_endpoints["qwen"], json=json_qwen) as q:
            q.raise_for_status()
            data = await q.json()
            response = data["response"]
            logger.info("qwen is good")
            return response
        
    except aiohttp.ClientError as e:
        logger.error(f"qwen failed: {e}")
        raise Exception(f"Failed at qwen: {e}")
    

async def send_llama(session, prompt):
    logger.info("Starting llama")

    json_llama = {
        "model": models["llama"],
        "prompt": prompt,
        "stream": False
    }

    try:
        logger.info("Hit llama API")
        async with session.post(api_endpoints["llama"], json=json_llama) as ll:
            ll.raise_for_status()
            data = await ll.json()
            response = data["response"]
            logger.info("llama good")
            return response
    
    except aiohttp.ClientError as e:
        logger.error(f"llama failed: {e}")
        raise Exception(f"Failed at llama: {e}")
    

async def send_all_models(session, user_input):
    logger.info("Start gather")
    
    optimized_prompt = await promptimizer(session, user_input)

    send = await asyncio.gather(
        send_qwen_small(session, optimized_prompt),
        send_qwen(session, optimized_prompt),
        send_llama(session, optimized_prompt),
        return_exceptions = True
    )

    for i, s in enumerate(send):
        if isinstance(s, Exception):
            logger.error(f"Model {i} returned error: {s}")
            raise s
    
    logger.info("Gather worked")
    return send[0], send[1], send[2]


async def send_judge(session, user_input, qwen_small_answer, llama_answer, qwen_answer):
    """Have the judge model select the best answer."""
    logger.info("STEP 3: Starting judge")
    
    judge_prompt = f"""You are an impartial judge evaluating three AI model responses to a user query.\n
    Select the BEST response.

USER QUERY: {user_input}

Model Responses:{qwen_small_answer},{llama_answer},{qwen_answer}

EVALUATION CRITERIA (in order of importance):
1. CORRECTNESS: Is the information accurate and factually correct?
2. COMPLETENESS: Does it fully address the user's query?
3. CLARITY: Is it well-structured and easy to understand?
4. CONCISENESS: Is it appropriately detailed without unnecessary fluff?

INSTRUCTIONS: Output ONLY the complete text of the best response. Do not include any labels,\n
explanations, or commentary about your choice."""

    json_judge = {
        "model": models["judge"],
        "prompt": judge_prompt,
        "stream": False
    }

    try:
        logger.info("Calling judge API...")
        async with session.post(api_endpoints["judge"], json=json_judge) as jud:
            jud.raise_for_status()
            data = await jud.json()
            response = data["response"]
            logger.info("Judge worked")
            return str(response)
        
    except aiohttp.ClientError as e:
        logger.error(f"Judge failed: {e}")
        raise Exception(f"Failed at judge: {e}")
    

async def main():

    logger.info("Gork will make his decision shortly...")

    print("\nYou now have the pleasure of speaking with Gork,")
    print("the world's closest attempt to AGI.")
    print("Type 'exit' to quit.\n")
    sys.stdout.flush()

    # Create session with no timeout - models can take as long as they need
    timeout = aiohttp.ClientTimeout(total=None, connect=None, sock_read=None, sock_connect=None)
    async with aiohttp.ClientSession(timeout=timeout) as session:

        
        while True:
            try:
                user_input = await asyncio.get_running_loop().run_in_executor(
                    None, lambda: input("YOU: ")
                )

                if user_input.lower() == "exit":
                    logger.info("But I'm better than ChatGPT, right?")
                    break

                logger.info(f"Received input: {user_input}")
                
                qwen_small_response, qwen_response, llama_response = await send_all_models(session, user_input)
                reply = await send_judge(session, user_input, qwen_small_response, llama_response, qwen_response)

                print(f"\nReply: {str(reply)}\n")
                sys.stdout.flush()

            except Exception as failed:
                logger.error(f"ERROR in main loop: {failed}", exc_info=True)
                print(f"\nError: {failed}\n")
                sys.stdout.flush()


if __name__ == "__main__":
    logger.info("Script started")
    asyncio.run(main())
