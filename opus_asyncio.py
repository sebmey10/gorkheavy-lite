import asyncio
import aiohttp
import json
"""gc.collect in order to clear unwanted memory when I want to."""

# These are the sockets of each container that I'm going to deploy.
api_endpoints = {
    "promptimizer_granite": "http://promptimizer:11434/api/generate",
    "llama": "http://llama:11434/api/generate",
    "qwen": "http://llama:11434/api/generate",
    "qwen_small": "http://qwen-small:11434/api/generate",
    "judge": "http://judge:11434/api/generate",
}

# These are the models I'm using to execute the workflow
models = {
    "promptimizer": "granite4:350m",
    "llama": "llama3.2:1b-instruct-q4_0",
    "qwen": "qwen2.5-coder:1.5b-instruct-q4_0",
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



    



# import aiohttp
# import asyncio

# # These are the sockets of each container that I'm going to deploy.
# api_endpoints = {
#     "promptimizer": "http://promptimizer:11434/api/generate",
#     "llama": "http://llama:11434/api/generate",
#     "qwen": "http://qwen:11434/api/generate",
#     "qwen_small": "http://qwen-small:11434/api/generate",
#     "judge": "http://judge:11434/api/generate",
# }

# # These are the models I'm using to execute the workflow
# models = {
#     "promptimizer": "granite4:350m",
#     "llama": "llama3.2:1b",
#     "qwen": "qwen2.5-coder:1.5b",
#     "qwen_small": "qwen3:0.6b",
#     "judge": "gemma3:1b"
# }

# llama_logfile = []
# qwen_logfile = []
# qwen_small_logfile = []


# async def promptimizer(session, user_input):
#     """Optimize the user's prompt for better model responses."""
#     promptimizer_prompt = f"""
#     Take {user_input} and rewrite it into a more concise query. The goal is to provide AI systems with a clear, 
#     focused prompt for optimal interpretation and response. Only respond with the re-written query."""

#     json_promptimizer = {
#         "model": models["promptimizer"],
#         "prompt": promptimizer_prompt,
#         "stream": False
#     }

#     try:
#         async with session.post(api_endpoints["promptimizer"], json=json_promptimizer) as response:
#             response.raise_for_status()
#             response_data = await response.json()
#             message_promptimizer = response_data["response"]
#             return message_promptimizer

#     except aiohttp.ClientError as failed:
#         raise Exception(f"Promptimizer didn't work: {str(failed)}")


# async def call_qwen_small(session, optimized_prompt):
#     """Call the Qwen Small model."""
#     json_qwen_small = {
#         "model": models["qwen_small"],
#         "prompt": optimized_prompt,
#         "stream": False
#     }
    
#     try:
#         async with session.post(api_endpoints["qwen_small"], json=json_qwen_small) as response:
#             response.raise_for_status()
#             response_data = await response.json()
#             message_qwen_small = response_data["response"]
#             qwen_small_logfile.append({"role": "assistant", "content": message_qwen_small})
#             return message_qwen_small
            
#     except aiohttp.ClientError as failed:
#         raise Exception(f"Qwen Small didn't work: {str(failed)}")


# async def call_llama(session, optimized_prompt):
#     """Call the LLaMA model."""
#     json_llama = {
#         "model": models["llama"],
#         "prompt": optimized_prompt,
#         "stream": False
#     }

#     try:
#         async with session.post(api_endpoints["llama"], json=json_llama) as response:
#             response.raise_for_status()
#             response_data = await response.json()
#             message_llama = response_data["response"]
#             llama_logfile.append({"role": "assistant", "content": message_llama})
#             return message_llama
            
#     except aiohttp.ClientError as failed:
#         raise Exception(f"LLaMA didn't work: {str(failed)}")


# async def call_qwen(session, optimized_prompt):
#     """Call the Qwen model."""
#     json_qwen = {
#         "model": models["qwen"],
#         "prompt": optimized_prompt,
#         "stream": False
#     }

#     try:
#         async with session.post(api_endpoints["qwen"], json=json_qwen) as response:
#             response.raise_for_status()
#             response_data = await response.json()
#             message_qwen = response_data["response"]
#             qwen_logfile.append({"role": "assistant", "content": message_qwen})
#             return message_qwen
            
#     except aiohttp.ClientError as failed:
#         raise Exception(f"Qwen didn't work: {str(failed)}")


# async def send_message_models(session, user_input):
#     """Optimize prompt, then call all three models in parallel."""
#     # First, optimize the prompt
#     optimized_prompt = await promptimizer(session, user_input)
    
#     # Then call all three models in parallel for faster execution
#     results = await asyncio.gather(
#         call_qwen_small(session, optimized_prompt),
#         call_llama(session, optimized_prompt),
#         call_qwen(session, optimized_prompt),
#         return_exceptions=True
#     )
    
#     # Check for exceptions
#     for result in results:
#         if isinstance(result, Exception):
#             raise result
    
#     return results[0], results[1], results[2]


# async def make_judgement(session, user_input):
#     """Have the judge model select the best answer."""
#     judge_prompt = f"""
#     User query: {user_input}

#     qwen_small answer: {qwen_small_logfile[-1]['content']}
#     LLaMA answer: {llama_logfile[-1]['content']}
#     Qwen answer: {qwen_logfile[-1]['content']}

#     Choose the best answer based on correctness, completeness, clarity, and usefulness.
#     Return the contents of the best answer, nothing else.
#     """

#     json_judge = {
#         "model": models["judge"],
#         "prompt": judge_prompt,
#         "stream": False
#     }

#     try:
#         async with session.post(api_endpoints["judge"], json=json_judge) as response:
#             response.raise_for_status()
#             response_data = await response.json()
#             message_judge = response_data["response"]
#             return str(message_judge)
    
#     except aiohttp.ClientError as failed:
#         raise Exception(f"Judge didn't work: {str(failed)}")


# async def main():
#     print("Chatting with gorkheavy-lite! (type 'exit' to quit)")
    
#     # Create a single session for all requests with extended timeout for slow models
#     timeout = aiohttp.ClientTimeout(total=300)  # 5 minute timeout
#     async with aiohttp.ClientSession(timeout=timeout) as session:
#         while True:
#             # Get user input in async-friendly way
#             user_input = await asyncio.get_event_loop().run_in_executor(
#                 None, lambda: input("YOU: ").strip()
#             )

#             if user_input.lower() == "exit":
#                 print("Bye!")
#                 break

#             try:
#                 await send_message_models(session, user_input)
#                 reply = await make_judgement(session, user_input)
#                 print(f"Reply: {reply}\n")
#             except Exception as failed:
#                 print(f"Error: {failed}")


# if __name__ == "__main__":
#     asyncio.run(main())
