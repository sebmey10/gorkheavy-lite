import asyncio
import aiohttp
import sys
import signal
import os
from datetime import datetime

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

# Note: We're using lists instead of appending to maintain conversation history

conversation_memory = []
shutdown_flag = False

def log(message):
    """Log with timestamp for better debugging in container logs"""
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{timestamp}] {message}", flush=True)

def handle_shutdown(signum, frame):
    """Handle shutdown signals gracefully"""
    global shutdown_flag
    log(f"Received signal {signum}, initiating graceful shutdown...")
    shutdown_flag = True


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
        log("Sending prompt to promptimizer for optimization...")
        async with session.post(api_endpoints["promptimizer"], json=json_promptimizer) as response:
            response.raise_for_status()
            data = await response.json()
            message = data["response"]
            conversation_memory.append({"role": "promptimizer", "content": message})
            log("Promptimizer optimization complete")
            return message
    except aiohttp.ClientError as f:
        log(f"Promptimizer failed: {f}, using original input")
        return user_input
    

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
    log("Sending optimized prompt to all models (qwen_small, qwen, llama) in parallel...")

    send = await asyncio.gather(
        send_qwen_small(session, optimized_prompt),
        send_qwen(session, optimized_prompt),
        send_llama(session, optimized_prompt),
        return_exceptions = True
    )

    for s in send:
        if isinstance(s, Exception):
            raise s

    log("All models have responded")
    return send[0], send[1], send[2]


async def send_judge(session, user_input, qwen_small_answer, llama_answer, qwen_answer):

    context_memory = conversation_memory[-5:]
    """Have the judge model select the best answer."""
    judge_prompt = f"""
    User query: {user_input}

    qwen_small answer: {qwen_small_answer}
    LLaMA answer: {llama_answer}
    Qwen answer: {qwen_answer}

    Choose the best answer based on correctness, completeness, clarity, and usefulness.
    Return the contents of the best answer, nothing else.

    To reference context of the conversation you're having, reference {context_memory}.
    """

      # Limit to last 10 messages for context
    json_judge = {
        "model": models["judge"],
        "prompt": judge_prompt,
        "stream": False
    }


    try:
        log("Sending all responses to judge for final selection...")
        async with session.post(api_endpoints["judge"], json = json_judge) as jud:
            jud.raise_for_status()
            data = await jud.json()
            response = data["response"]
            conversation_memory.append({"role":"judge", "content": response})
            log("Judge has selected the best response")
            return str(response)

    except aiohttp.ClientError as e:
        raise Exception(f"Failed at judge: {e}")
    

async def read_input_with_timeout(prompt_text, timeout=30):
    """Read input with timeout to prevent hanging in containerized environments"""
    try:
        # Check if stdin is actually a TTY and connected
        if not sys.stdin.isatty():
            log("Warning: stdin is not a TTY, input may not work as expected")

        print(prompt_text, end="", flush=True)

        # Use a timeout to prevent indefinite hanging
        user_input = await asyncio.wait_for(
            asyncio.get_event_loop().run_in_executor(None, sys.stdin.readline),
            timeout=timeout
        )
        return user_input
    except asyncio.TimeoutError:
        return None
    except Exception as e:
        log(f"Error reading input: {e}")
        return None


async def main():
    global shutdown_flag

    # Register signal handlers for graceful shutdown
    signal.signal(signal.SIGTERM, handle_shutdown)
    signal.signal(signal.SIGINT, handle_shutdown)

    log("=" * 60)
    log("Gork AI Ensemble Orchestrator Starting")
    log("=" * 60)
    log(f"Python version: {sys.version}")
    log(f"stdin is TTY: {sys.stdin.isatty()}")
    log(f"stdin is connected: {not sys.stdin.closed}")

    # Check environment for test mode
    test_mode = os.environ.get("TEST_MODE", "").lower() == "true"
    if test_mode:
        log("Running in TEST MODE")

    log("\nYou now have the pleasure of speaking with Gork,")
    log("the world's closest attempt to AGI.")
    log("Type 'exit' to quit.\n")

    # Test all services are reachable
    log("Testing connectivity to AI model services...")
    async with aiohttp.ClientSession() as test_session:
        for service_name, endpoint in api_endpoints.items():
            try:
                # Just check the base URL is reachable
                base_url = endpoint.replace("/api/generate", "/api/tags")
                async with test_session.get(base_url, timeout=aiohttp.ClientTimeout(total=5)) as resp:
                    if resp.status == 200:
                        log(f"✓ {service_name} is reachable")
                    else:
                        log(f"⚠ {service_name} returned status {resp.status}")
            except Exception as e:
                log(f"✗ {service_name} is not reachable: {e}")

    log("\nOrchestrator ready! Waiting for input...")
    log("(In Kubernetes, use: kubectl attach -it <pod-name>)\n")
    sys.stdout.flush()

    async with aiohttp.ClientSession() as session:
        last_activity = datetime.now()

        while not shutdown_flag:
            try:
                # Send periodic keepalive message
                elapsed = (datetime.now() - last_activity).total_seconds()
                if elapsed > 60:
                    log(f"Orchestrator is alive and waiting for input... (uptime: {int(elapsed)}s)")
                    last_activity = datetime.now()

                # Try to read input with timeout
                user_input = await read_input_with_timeout("YOU: ", timeout=30)

                # Handle various input scenarios
                if user_input is None:
                    # Timeout or error - just continue
                    await asyncio.sleep(1)
                    continue

                if not user_input or user_input.strip() == "":
                    # Empty input
                    await asyncio.sleep(1)
                    continue

                user_input = user_input.strip()
                last_activity = datetime.now()

                if user_input.lower() == "exit":
                    log("Exit command received, shutting down...")
                    break

                log(f"Received input: {user_input}")

                # Add user input to memory so the judge has context
                conversation_memory.append({"role": "user", "content": user_input})

                # Keep memory small (rolling window of last 20 messages)
                if len(conversation_memory) > 20:
                    conversation_memory = conversation_memory[-20:]

                qwen_small_response, qwen_response, llama_response = await send_all_models(session, user_input)
                reply = await send_judge(session, user_input, qwen_small_response, llama_response, qwen_response)

                print(f"\nGORK: {str(reply)}\n", flush=True)
                log("Response delivered successfully")

            except Exception as failed:
                log(f"Error processing request: {failed}")
                import traceback
                log(f"Traceback: {traceback.format_exc()}")
                await asyncio.sleep(1)

    log("Orchestrator shutting down gracefully...")
    log("Goodbye!")

                


if __name__ == "__main__":
    try:
        log("Starting Gork AI Orchestrator...")
        asyncio.run(main())
    except KeyboardInterrupt:
        log("Received keyboard interrupt, shutting down...")
    except Exception as e:
        log(f"Fatal error: {e}")
        import traceback
        log(f"Traceback: {traceback.format_exc()}")
        sys.exit(1)
