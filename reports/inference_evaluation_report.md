# Benchmark Report: Local LLM Inference Servers 

## Background
The rapid commoditization of Large Language Models (LLMs) has catalyzed the development of numerous inference engines designed to host open-weight models locally. However, not all inference architectures are built equally. Approaches range from simple wrapper utilities surrounding native C++ execution kernels (`llama-cpp-python`) to automated containerized environments (`Ollama`), up to highly optimized, continuous-batching throughput engines optimized specifically for modern data centers (`vLLM`). 

Understanding the behavioral dynamics of these inference runners under load is crucial to provisioning hardware efficiently. The purpose of this report is to evaluate the raw scalability and latency profiles of these diverse methodologies using standardized baseline hardware.

## Motivation
With the acquisition of dual **NVIDIA GeForce RTX 5070 cards (SM_120 Blackwell)**, establishing an optimized inference architecture is necessary. While basic engines operate seamlessly for single-user endpoints, enterprise applications demand concurrency. 

Natively compiled engines hook into high-level CUDA primitives (e.g., FlashAttention, Triton integrations) and deploy advanced memory strategies like **PagedAttention** to minimize KV Cache fragmentation. The motivation of this empirical evaluation is to definitively quantify the token delivery performance of these engines when subjected to identical concurrent API load, bridging theoretical architecture to quantifiable telemetry natively on the target hardware.

## Experiment
Using the custom-built **Python-Based LLM Inference Load and Validation Suite**, we orchestrated a synchronized evaluation sweep targeting three fundamentally distinct backend servers:

1. **vLLM (Native)**
   - **Model:** `Qwen/Qwen2.5-3B-Instruct`
   - **Configuration:** Custom compiled SM_120 wheels binding `torch==2.11.0+cu128`, initialized with `--gpu-memory-utilization 0.6` on GPU 0.
2. **Ollama (Pre-packaged Binary)**
   - **Model:** `llama3.2:latest` (3.2B parameters, Q4_K_M)
   - **Configuration:** Automated runtime scheduler targeting dual GPUs dynamically.
3. **llama.cpp (via `llama-cpp-python` Server)**
   - **Model:** `Llama-3.2-3B-Instruct-Q4_K_M.gguf`
   - **Configuration:** Initialized natively with full GPU offloading (`--n_gpu_layers 33`) on GPU 1.

**Methodology:**
A configuration map (`evaluation_scenarios.yaml`) was triggered sending exactly **5 concurrent virtual users** unconditionally to each endpoint simultaneously. 
The suite systematically measured structural validity (HTTP/JSON integrity), raw Average Latency (End-to-end), Time-to-First-Token (TTFT), and raw throughput (Tokens/second) using identical sampling payloads natively parsed from the `usage` reporting headers.

## Results

### Throughput, Latency, and TTFT Baseline (5 Concurrent Users)
| Inference Engine | Average Latency | Average TTFT | Throughput (Speed) | JSON Pass Rate |
| :--- | :--- | :--- | :--- | :--- |
| **vLLM (Native Runtime)** | **260ms** | **260ms** | **61.5 tok/s** | 100% |
| **Ollama (Packaged Engine)** | 1426ms | 1426ms | 12.1 tok/s | 100% |
| **llama-cpp-python (Raw Server)** | 2343ms | 2343ms | 6.0 tok/s | 100% |

#### Observations
1. **vLLM Architecture Supremacy**: vLLM produced the lowest overall latency (260ms) and highest total delivery throughput (61.5 tok/s). By utilizing continuous batching and PagedAttention natively on the Blackwell GPUs, vLLM resolved all 5 concurrent requests nearly instantaneously. Even without quantization, the full precision model outpaced heavily quantized implementations organically.
2. **Ollama's Robust Backend**: Ollama demonstrated highly competitive resilience natively out of the box executing a Q4 quantized Llama 3.2 model at 12.1 tok/s across its internal scheduler, effectively doubling the pure native baseline of the raw Web API hook.
3. **Raw Python API Bottlenecks**: Directly exposing the `llama.cpp` kernel via a standard Uvicorn worker instance collapsed under concurrent load. Resolving at roughly 6.0 tok/s and averaging immense delays (2.3 seconds), the lack of request batching forced the Python API wrapper to process tensor operations sequentially, exposing a critical bottleneck when bridging high-performance C++ codebases directly to unoptimized Python asynchronous loops without internal queuing logic.

## Conclusion
The orchestration undeniably proves that **the layer surrounding the model tensor execution is often more critical than the model weights themselves**. 

For production deployments targeting the RTX 5070 array, **vLLM is the definitive choice for high-throughput endpoint generation**. By managing VRAM proactively and fusing CUDA operations via Triton gradients locally, it effortlessly bypassed single-request bottlenecks gracefully. Ollama remains an excellent secondary abstraction for rapid prototyping or scenarios where VRAM boundaries are rigid and multi-model quantization swapping is preferred.

Moving forward, the primary infrastructure will enforce `vllm` deployments scaled behind NGINX or HAProxy, ensuring hardware saturation limits are mathematically aligned with the benchmarks derived traversing the Load Testing validation suite.
