# GorkHeavy-Lite

An AI ensemble system that combines multiple models to deliver intelligent responses.

## Overview

GorkHeavy-Lite orchestrates multiple AI models working together:
- **Promptimizer** optimizes user queries
- **3 AI models** (Qwen Small, Qwen, Llama) generate responses in parallel
- **Judge model** selects the best answer

## Deployment Options

### Local Docker (Recommended)
```bash
cd local_docker
docker-compose -f local_docker.yaml up -d
docker exec -it gork-orchestrator python local_docker.py
```

### Kubernetes (k3s)
```bash
kubectl apply -f k3s/script_final.yaml
kubectl exec -it <orchestrator-pod> -- python final_script.py
```

### Llama Swap
```bash
python llama_swap/swap_script.py
```

## Models Used

- Promptimizer: Granite 4 350M
- Qwen Small: Qwen 3 0.6B
- Qwen: Qwen 2.5 Coder 1.5B
- Llama: Llama 3.2 1B
- Judge: Gemma 3 1B

## Features

- Async processing with asyncio
- Parallel model execution
- Automatic best response selection
- Multiple deployment options
- Fault tolerance

## Usage

Type your query when prompted. Enter 'exit' to quit.
