# Gork Heavy-Lite

An AI ensemble system that orchestrates multiple Large Language Models (LLMs) working together to provide optimized responses through parallel processing and intelligent selection.

## Overview

Gork Heavy-Lite is a distributed AI architecture that leverages the strengths of multiple small language models working in concert. Rather than relying on a single large model, the system:

1. **Optimizes** user prompts using a specialized model
2. **Processes** queries in parallel across multiple LLM models
3. **Evaluates** responses using a judge model
4. **Selects** the best response from the ensemble

This approach combines the benefits of different model architectures and sizes while optimizing for response quality and efficiency.

## Architecture

```
User Input
    ↓
Promptimizer (Granite 4 350M) - Optimizes the prompt
    ↓
┌───────────────────┬───────────────────┬──────────────────┐
│  Qwen Small 0.6B  │  Qwen 1.5B Coder  │  LLaMA 3.2 1B   │
│  (Fast Response)  │  (Code-Focused)   │  (General)      │
└───────────────────┴───────────────────┴──────────────────┘
    ↓                       ↓                     ↓
                    Judge (Gemma 3 1B)
                            ↓
                    Best Response Selected
```

## Features

- **Asynchronous Processing**: Built with Python `asyncio` for efficient concurrent operations
- **Ensemble Voting**: Multiple models generate responses in parallel for quality optimization
- **Intelligent Selection**: Judge model evaluates and selects the best response
- **Kubernetes Native**: Designed to run on k3s with proper resource management
- **Distributed Architecture**: Models spread across nodes with topology constraints
- **Production Ready**: Includes health checks, resource limits, and signal handling

## Prerequisites

- **Kubernetes**: k3s cluster (or compatible Kubernetes distribution)
- **kubectl**: Kubernetes command-line tool
- **Docker**: For building container images (optional if using pre-built images)
- **Hardware**: Sufficient resources for running 5 concurrent LLM models

### Minimum Resource Requirements

- **Memory**: ~5GB total (across all models)
- **CPU**: ~2.5 cores total
- **Storage**: ~10GB for model weights and containers

## Deployment

### Quick Start

1. **Deploy to Kubernetes**:
   ```bash
   kubectl apply -f script_final.yaml
   ```

2. **Verify Deployment**:
   ```bash
   kubectl get pods
   kubectl logs -f deployment/final-script
   ```

3. **Access the System**:
   ```bash
   kubectl port-forward deployment/final-script 8080:8080
   ```

### Building from Source

If you need to build the Docker images:

```bash
# Build orchestrator
docker build -f Dockerfile.final_script -t final-script:latest .

# Build models
docker build -f Dockerfile.judge -t judge:latest .
docker build -f Dockerfile.llama -t llama:latest .
docker build -f Dockerfile.promptimizer -t promptimizer:latest .
docker build -f Dockerfile.qwen -t qwen:latest .
docker build -f Dockerfile.qwen_small -t qwen-small:latest .
```

## Usage

Once deployed, the system provides an interactive CLI interface:

1. Enter your query when prompted
2. The system will:
   - Optimize your prompt
   - Query multiple models in parallel
   - Have a judge select the best response
3. View the selected response
4. Type `exit` to quit

### Example Interaction

```
Enter your query (or 'exit' to quit): What is quantum computing?

[System processes query through ensemble]

Response: [Judge's selected best answer from the three models]
```

## Project Structure

```
gorkheavy-lite/
├── final_script.py              # Main orchestrator (async Python)
├── script_final.yaml            # Kubernetes deployment manifest
├── requirements.txt             # Python dependencies
├── Dockerfile.final_script      # Orchestrator container
├── Dockerfile.judge             # Judge model container
├── Dockerfile.llama             # LLaMA model container
├── Dockerfile.promptimizer      # Promptimizer container
├── Dockerfile.qwen              # Qwen model container
└── Dockerfile.qwen_small        # Qwen small model container
```

## Model Ensemble

| Service | Model | Size | Purpose |
|---------|-------|------|---------|
| **Promptimizer** | Granite 4 | 350M | Optimize and rewrite user prompts |
| **Qwen Small** | Qwen 3 | 0.6B | Fast response generation |
| **Qwen** | Qwen 2.5-Coder | 1.5B | Code-focused response generation |
| **LLaMA** | LLaMA 3.2 | 1B | General-purpose response generation |
| **Judge** | Gemma 3 | 1B | Evaluate and select best response |

## Technologies

- **Language**: Python 3.11
- **Async Framework**: `asyncio`
- **HTTP Client**: `aiohttp` 3.9.1
- **LLM Runtime**: Ollama
- **Container Orchestration**: Kubernetes (k3s)
- **Container Runtime**: Docker

## How It Works

1. **User Input**: The orchestrator accepts user queries through an interactive CLI
2. **Prompt Optimization**: Query is sent to the Promptimizer to enhance/rewrite
3. **Parallel Processing**: Optimized prompt is dispatched to three models simultaneously:
   - Qwen Small (fastest, 0.6B parameters)
   - Qwen Coder (code-specialized, 1.5B parameters)
   - LLaMA (general-purpose, 1B parameters)
4. **Response Collection**: All three responses are gathered asynchronously
5. **Judging**: The Judge model evaluates all responses and selects the best one
6. **Output**: Selected response is returned to the user

### Conversation Memory

The system maintains conversation history using efficient list-based memory management, allowing for context-aware multi-turn conversations.

## Configuration

### Resource Limits

Default resource allocations (configurable in `script_final.yaml`):

- **Models**: 512Mi-2Gi memory, 250m-900m CPU per model
- **Orchestrator**: 128Mi memory, 100m CPU

### Deployment Constraints

- **Topology Spread**: Ensures models distribute across available nodes
- **Node Affinity**: Prevents deployment on Kubernetes control-plane nodes
- **Init Containers**: Verify all model services are ready before starting orchestrator

## Development

### Local Development

```bash
# Install dependencies
pip install -r requirements.txt

# Run locally (requires Ollama and models installed)
python final_script.py
```

### Modifying Models

To use different models, update the corresponding Dockerfile with the desired Ollama model:

```dockerfile
RUN ollama run <model-name>
```

## Contributing

This project demonstrates an ensemble AI architecture. Contributions, suggestions, and improvements are welcome.

## License

[Add your license here]

## Acknowledgments

- Built with [Ollama](https://ollama.ai/) for local LLM inference
- Uses models from Meta (LLaMA), Alibaba (Qwen), Google (Gemma), and IBM (Granite)

---

**Note**: This is a research/demonstration project showcasing distributed AI architecture and ensemble learning techniques.
