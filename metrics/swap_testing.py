import asyncio
import aiohttp
import sys
import logging
import ollama
from ollama import chat, ChatResponse

# FLOW OVERVIEW (numbered in execution order)
# 1) Python imports modules (top of file)
# 2) Logging configured (logging.basicConfig)
# 3) Endpoint + model registry defined
# 4) main() is launched via asyncio.run(main())
# 5) main() opens ONE aiohttp ClientSession
# 6) Loop: read user input
# 7) promptimizer() rewrites input into optimized_prompt
# 8) send_all_models() gathers candidate model responses concurrently
# 9) send_judge() selects best answer and returns it
# 10) main() prints reply and loops back
# 11) On 'exit': cleanup_models() attempts to unload models, then quits
 
 
 # Note: Fix statelessness of LLM's at end of script

 # Getting the flow

# 2) Logging setup happens at import-time (before main())
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    stream=sys.stdout
)
logger = logging.getLogger(__name__)

# 3) API endpoint (OpenAI-compatible schema: /v1/chat/completions)
ollama_endpoint = "http://hal.kub.org:8080/v1/chat/completions"
 
 
# These are the models I'm using to execute the workflow
# 3.1) Central registry: add a model here to make it available to the script

models = {
    "promptimizer": "Olmo-3-7B-Instruct-Q8_0",
    "llama": "llama3.1-8B-Q8_0",
    "gemma": "gpt-oss-20b-F16",
    "gemma_small": "Olmo-3-7B-Think-Q8_0",
    "judge": "granite-3.2-8b-instruct-f16"
}

OTHER_KEYS = {"promptimizer", "judge"}

# 3.2) Candidate model keys = everything except the role models above
CANDIDATE_KEYS = [i for i in models.keys() if i not in OTHER_KEYS]

    # json_promptimizer = {
    #     "model": models["promptimizer"],
    #     "messages": [{"role": "user", "content":prompt_text}],
    #     "keep_alive": -1
    # } Reference structure for the below
 
#     {
#   "model": "llama3",
#   "messages": [
#     { 
#       "role": "system", 
#       "content": "You are a helpful assistant. You explain things simply but skip the fluff. Use a supportive but direct tone." 
#     },
#     { 
#       "role": "user", 
#       "content": "Hi, I'm Jayden." 
#     },
#     { 
#       "role": "assistant", 
#       "content": "Hello Jayden! How can I help you today?" 
#     },
#     { 
#       "role": "user", 
#       "content": "Explain 'statelessness' in one sentence." 
#     }
#   ],
#   "stream": false
# } <- response structure. This grows as conversation grows. User sends something, LLM sends back. The ONLY way the LLM knows is via the repsonse, it does not hold context. Right now our models are stateless

def build_payload(model_key: str,*,messages: list[dict[str, str]], keep_alive: int = -1) -> dict:  # Note * is key=value or name=value type hint, messages are in "messages": [{"role": "user", "content":prompt_text}] format, int default is -1
    """
    Build the exact JSON body sent to /v1/chat/completions.

    note: the `*` means `messages=` and `keep_alive=` must be passed by keyword (similar to **kwargs but for type hints, non compulsory).
    """
    
    return {
        "model": models[model_key],
        "messages": messages,
        "keep_alive": keep_alive,
    }

# 3.3) NOTE: The next two defs are placeholders/stubs right now.
#      They are not part of the current running flow because:
#      - call_model() is incomplete and will syntax-error if executed
#      - send_all_models() is defined later again (that later one is used)

# I'll use build payload within initial model call
async def call_model(session: aiohttp.ClientSession ):
    "Generic model caller for scalable response per what we put in models (dict)"
# Key for all models setup here
    for key in CANDIDATE_KEYS:
        payload = build_payload(key,messages=[{"role": "user", "content":}])
    async with session.post(ollama_endpoint,json=) # Put the different jsons in this structure

def send_all_models():
    pass


async def promptimizer(session, user_input):
    # 7) Step: Promptimizer
    # 7.1) Build the promptimizer instruction text
    logger.info("Starting promptimizer")
   
    prompt_text = f"""You are an expert prompt engineer. Transform the following user request into an optimal prompt that will produce the best possible response from an AI model.
 
USER REQUEST: {user_input}
 
OPTIMIZATION RULES:
1. Preserve the original intent completely
2. Add specificity: define scope, format, and constraints
3. Request step-by-step reasoning for complex tasks
4. Specify the desired output format (e.g., bullet points, code, explanation)
5. Remove ambiguity while keeping the prompt concise
IMPORTANT: Tell models that they have to give a concise response.
 
OUTPUT: Return ONLY the optimized prompt with no preamble, explanation, or meta-commentary."""
 
    # 7.2) Build JSON payload for promptimizer
    json_promptimizer = {
        "model": models["promptimizer"],
        "messages": [{"role": "user", "content":prompt_text}],
        "keep_alive": -1
    }
 
    try:
        # 7.3) POST to endpoint, parse JSON, extract assistant content
        logger.info("Hit promptimizer API")
        async with session.post(ollama_endpoint, json=json_promptimizer) as response:
            response.raise_for_status()
            data = await response.json()
            logger.info(data)
            message = data["choices"][0]["message"]["content"]
            logger.info("Promptimizer is good")
            return message
    except aiohttp.ClientError as f:
        # 7.4) On failure, fall back to original user_input
        logger.error(f"Promptimizer failed: {f}")
        return user_input
   
 
async def send_qwen_small(session, prompt):
    # 8.1) Candidate model call: gemma_small
    logger.info("Start qwen_small")
 
    json_qwen_small = {
        "model": models["gemma_small"],
        "messages": [{"role": "user", "content": prompt}],
        "keep_alive": -1
        }
 
    try:
        logger.info("Hit qwen_small API")
        async with session.post(ollama_endpoint, json=json_qwen_small) as qs:
            qs.raise_for_status()
            data = await qs.json()
            logger.info(data)
            message = data["choices"][0]["message"]["content"]
            logger.info("qwen_small is good")
            return message
    except aiohttp.ClientError as e:
        logger.error(f"qwen_small failed: {e}")
        raise Exception(f"Failed at qwen_small: {e}")
   
 
async def send_qwen(session, prompt):
    # 8.2) Candidate model call: gemma
    logger.info("Starting qwen")
   
    json_qwen = {
        "model": models["gemma"],
        "messages": [{"role": "user", "content": prompt}],
        "keep_alive": -1
    }
 
    try:
        logger.info("Hit qwen API...")
        async with session.post(ollama_endpoint, json=json_qwen) as q:
            q.raise_for_status()
            data = await q.json()
            response = data["choices"][0]["message"]["content"]
            logger.info("qwen is good")
            return response
       
    except aiohttp.ClientError as e:
        logger.error(f"qwen failed: {e}")
        raise Exception(f"Failed at qwen: {e}")
   
 
async def send_llama(session, prompt):
    # 8.3) Candidate model call: llama
    logger.info("Starting llama")
 
    json_llama = {
        "model": models["llama"],
        "messages": [{"role": "user", "content": prompt}],
        "keep_alive": -1
    }
 
    try:
        logger.info("Hit llama API")
        async with session.post(ollama_endpoint, json=json_llama) as ll:
            ll.raise_for_status()
            data = await ll.json()
            response = data["choices"][0]["message"]["content"]
            logger.info("llama good")
            return response
   
    except aiohttp.ClientError as e:
        logger.error(f"llama failed: {e}")
        raise Exception(f"Failed at llama: {e}")
   
 
async def send_all_models(session, user_input):
    # 8) Step: run all candidate models concurrently
    # 8.0) First optimize the prompt (one time)
    logger.info("Start gather")
   
    optimized_prompt = await promptimizer(session, user_input)
 
    # 8.1) Then gather candidate outputs concurrently
    send = await asyncio.gather(
        send_qwen_small(session, optimized_prompt),
        send_qwen(session, optimized_prompt),
        send_llama(session, optimized_prompt),
        return_exceptions = True
    )
 
    # 8.2) If any candidate returned an Exception, raise it
    for i, s in enumerate(send):
        if isinstance(s, Exception):
            logger.error(f"Model {i} returned error: {s}")
            raise s
   
    # 8.3) Return responses in the same order they were gathered
    logger.info("Gather worked")
    return send[0], send[1], send[2]
 
 
async def send_judge(session, user_input, qwen_small_answer, llama_answer, qwen_answer):
    """Have the judge model select the best answer."""
    # 9) Step: Judge compares candidate outputs and chooses best
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
 
INSTRUCTIONS: Output ONLY the complete text of the best response as well as which model gave the best answer. Do not include any labels,\n
explanations, or commentary about your choice."""
 
    # 9.1) Build JSON payload for judge
    json_judge = {
        "model": models["judge"],
        "messages": [{"role": "user", "content": judge_prompt}],
        "keep_alive": -1
    }
 
    try:
        # 9.2) POST to endpoint, parse JSON, extract assistant content
        logger.info("Calling judge API...")
        async with session.post(ollama_endpoint, json=json_judge) as jud:
            jud.raise_for_status()
            data = await jud.json()
            response = data["choices"][0]["message"]["content"]
            logger.info("Judge worked")
            return str(response)
       
    except aiohttp.ClientError as e:
        logger.error(f"Judge failed: {e}")
        raise Exception(f"Failed at judge: {e}")
   
 
async def cleanup_models(session):
    # 11) Step: attempt to unload models on exit
    logger.info("Cleaning up models")
 
   
    for model_name, model_id in models.items():
        try:
            request_data = {
                "model": model_id,
                "keep_alive": 0
            }
 
            async with session.post(ollama_endpoint,json=request_data) as req:
                req.raise_for_status()
                data = await req.json()
               
               
        except Exception as e:
            logger.error(f"Error unloading {model_name}: {e}")
   
   
 
async def main():
 
    # 4) main() is the runtime entrypoint called by asyncio.run(main())
    logger.info("Gork will make his decision shortly...")
 
    print("\nYou now have the pleasure of speaking with Gork,")
    print("the world's closest attempt to AGI.")
    print("Type 'exit' to quit.\n")
    sys.stdout.flush()
 
    # 5) Create a single ClientSession for the entire chat loop
    timeout = aiohttp.ClientTimeout(total=None, connect=None, sock_read=None, sock_connect=None)
    async with aiohttp.ClientSession(timeout=timeout) as session:
 
       
        while True:
            try:
                # 6) Read user input without blocking event loop
                user_input = await asyncio.get_running_loop().run_in_executor(
                    None, lambda: input("YOU: ")
                )
 
                # 6.1) Exit path
                if user_input.lower() == "exit":
                    logger.info("But I'm better than ChatGPT, right?")
                    await cleanup_models(session)
                    break
 
                logger.info(f"Received input: {user_input}")
               
                # 7-8) promptimizer + gather candidates
                qwen_small_response, qwen_response, llama_response = await send_all_models(session, user_input)

                # 9) judge best response
                reply = await send_judge(session, user_input, qwen_small_response, llama_response, qwen_response)
 
                # 10) Print and loop
                print(f"\nReply: {str(reply)}\n")
                sys.stdout.flush()
 
            except Exception as failed:
                logger.error(f"ERROR in main loop: {failed}", exc_info=True)
                print(f"\nError: {failed}\n")
                sys.stdout.flush()
 
 
if __name__ == "__main__":
    # 4) Script entrypoint
    logger.info("Script started")
    asyncio.run(main())
