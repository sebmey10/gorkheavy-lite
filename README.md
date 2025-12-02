# Gork AI - Lightweight Ensemble AI Orchestrator

A lightweight, Kubernetes-native AI ensemble system that combines multiple small language models to provide intelligent responses. The system uses an orchestration pattern where multiple models generate responses in parallel, and a judge model selects the best answer.

## Architecture

The system consists of 6 main components:

1. **Promptimizer** (Granite 4 350m) - Optimizes and refines user queries for better model comprehension
2. **LLaMA** (LLaMA 3.2 1B) - General-purpose language model
3. **Qwen** (Qwen 2.5 Coder 1.5B) - Coding-specialized language model
4. **Qwen Small** (Qwen 3 0.6B) - Fast, lightweight model for quick responses
5. **Judge** (Gemma 3 1B) - Evaluates all responses and selects the best one
6. **Orchestrator** - Python asyncio-based coordinator that manages the workflow

## How It Works

```
User Input
    ↓
Promptimizer (optimizes query)
    ↓
┌───────────────┬───────────────┬───────────────┐
│   LLaMA 3.2   │     Qwen      │  Qwen Small   │  (parallel execution)
└───────────────┴───────────────┴───────────────┘
    ↓           ↓           ↓
           Judge Model
    (selects best response)
    ↓
User receives final answer
```

## Prerequisites

- Kubernetes cluster (tested on k3s)
- `kubectl` configured to access your cluster
- Docker images built and pushed to registry:
  - `sebastein/promptimizer:v5.0`
  - `sebastein/llama:v5.0`
  - `sebastein/qwen:v5.0`
  - `sebastein/qwen_small:v5.0`
  - `sebastein/judge:v5.0`
  - `sebastein/final_script:v6.0`

## Deployment

### 1. Deploy the System

```bash
kubectl apply -f script_final.yaml
```

This will deploy all 5 AI models and the orchestrator.

### 2. Wait for Pods to be Ready

```bash
kubectl get pods -w
```

Wait until all pods show `Running` status. The orchestrator pod has an init container that waits for all AI models to be ready before starting.

### 3. Check Orchestrator Logs

```bash
# Find the orchestrator pod name
kubectl get pods | grep orchestrator

# View logs
kubectl logs -f <orchestrator-pod-name>
```

You should see startup logs including:
- Service connectivity checks
- "Orchestrator ready! Waiting for input..."

## Usage

### Interactive Mode (Primary Method)

To interact with Gork AI:

```bash
# Find the orchestrator pod
kubectl get pods | grep orchestrator

# Attach to the pod
kubectl attach -it <orchestrator-pod-name>

# You should see:
# YOU:

# Type your question and press Enter
YOU: What is the capital of France?

# Gork will process your query through all models and respond
GORK: The capital of France is Paris...

# Type 'exit' to quit
YOU: exit
```

### Viewing Logs

All processing steps are logged with timestamps for debugging:

```bash
kubectl logs -f <orchestrator-pod-name>
```

You'll see logs like:
```
[2025-12-02 16:00:00] Orchestrator ready! Waiting for input...
[2025-12-02 16:00:15] Received input: What is the capital of France?
[2025-12-02 16:00:15] Sending prompt to promptimizer for optimization...
[2025-12-02 16:00:16] Promptimizer optimization complete
[2025-12-02 16:00:16] Sending optimized prompt to all models...
[2025-12-02 16:00:20] All models have responded
[2025-12-02 16:00:20] Sending all responses to judge...
[2025-12-02 16:00:22] Judge has selected the best response
[2025-12-02 16:00:22] Response delivered successfully
```

## Resource Requirements

### Minimum Requirements (per node)
- **CPU**: 2-3 cores
- **Memory**: 6-8 GB RAM
- **Storage**: 10 GB

### Individual Component Requirements

| Component    | CPU Request | CPU Limit | Memory Request | Memory Limit |
|--------------|-------------|-----------|----------------|--------------|
| Promptimizer | 100m        | 500m      | 256Mi          | 1Gi          |
| LLaMA        | 200m        | 800m      | 512Mi          | 1.5Gi        |
| Qwen         | 200m        | 900m      | 512Mi          | 2Gi          |
| Qwen Small   | 100m        | 500m      | 256Mi          | 1Gi          |
| Judge        | 200m        | 900m      | 512Mi          | 1.5Gi        |
| Orchestrator | 100m        | 200m      | 128Mi          | 256Mi        |

## Features

### Orchestrator Features
- ✅ Async/await for parallel model execution
- ✅ Automatic service health checks on startup
- ✅ Graceful signal handling (SIGTERM, SIGINT)
- ✅ Comprehensive timestamped logging
- ✅ Conversation memory (rolling window of 20 messages)
- ✅ Input timeout handling (prevents indefinite hangs)
- ✅ Periodic keepalive messages
- ✅ TTY detection and warnings
- ✅ Detailed error reporting with stack traces

### Kubernetes Features
- ✅ Init containers to wait for all services
- ✅ Service mesh with ClusterIP services
- ✅ Topology spread constraints for HA
- ✅ Node affinity (avoids control plane nodes)
- ✅ Liveness and readiness probes for AI models
- ✅ Resource limits and requests

## Troubleshooting

### Orchestrator not responding

```bash
# Check if the pod is running
kubectl get pods | grep orchestrator

# Check logs for errors
kubectl logs <orchestrator-pod-name>

# Check if all AI model services are ready
kubectl get pods
```

### Models taking too long to respond

The models need time to load. The init container waits 30 seconds after all services respond before starting the orchestrator. If you still have issues:

```bash
# Check individual model logs
kubectl logs <model-pod-name>

# Example:
kubectl logs $(kubectl get pods | grep llama | awk '{print $1}')
```

### "No input received (EOF)" errors

This is normal if you're viewing logs without being attached. The orchestrator will continue waiting for input.

### Cannot attach to orchestrator

Make sure the pod is in Running state:
```bash
kubectl get pods | grep orchestrator
```

If it's in CrashLoopBackOff, check the logs:
```bash
kubectl logs <orchestrator-pod-name>
```

## Building Docker Images

### Build Orchestrator Image

```bash
docker build -f Dockerfile.final_script -t sebastein/final_script:v6.0 .
docker push sebastein/final_script:v6.0
```

### Build Model Images

```bash
# Promptimizer
docker build -f Dockerfile.promptimizer -t sebastein/promptimizer:v5.0 .

# LLaMA
docker build -f Dockerfile.llama -t sebastein/llama:v5.0 .

# Qwen
docker build -f Dockerfile.qwen -t sebastein/qwen:v5.0 .

# Qwen Small
docker build -f Dockerfile.qwen_small -t sebastein/qwen_small:v5.0 .

# Judge
docker build -f Dockerfile.judge -t sebastein/judge:v5.0 .
```

## Cleanup

To remove all components:

```bash
kubectl delete -f script_final.yaml
```

## Technical Details

### Conversation Memory
The orchestrator maintains a conversation memory with a rolling window of 20 messages. The judge uses the last 5 messages for context when selecting the best response.

### Prompt Optimization
The promptimizer uses a specialized prompt engineering technique to:
1. Identify core intent
2. Remove unnecessary filler
3. Clarify constraints
4. Structure complex queries

### Error Handling
- Individual model failures are caught and logged
- If promptimizer fails, the original user input is used
- All exceptions include full stack traces in logs
- The orchestrator continues running even after errors

## Future Improvements

Potential enhancements:
- REST API endpoint for programmatic access
- Web UI for easier interaction
- Response caching for repeated queries
- Model performance metrics and tracking
- Dynamic model selection based on query type
- Streaming responses

## License

This project is open source. Feel free to use and modify.

## Contributing

Contributions are welcome! Please ensure:
- Code follows existing patterns
- All changes are tested in a k3s cluster
- Documentation is updated

## Support

For issues, please check:
1. Orchestrator logs: `kubectl logs <orchestrator-pod-name>`
2. Model logs: `kubectl logs <model-pod-name>`
3. Service status: `kubectl get services`
4. Pod status: `kubectl get pods`
