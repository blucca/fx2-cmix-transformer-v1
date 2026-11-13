## Janas test report

- Janas commit: `unknown`
- Levels: quick
- Model (quick): Qwen3-4B, run as `qwen3-4b-q4km.jns` - Janas's format (JNS), converted here by gguf2jns and jns_planes from `Qwen3-4B-Q4_K_M.gguf` of `Qwen/Qwen3-4B-GGUF`; converted in an earlier run, checked again by janas-get
- CPU: AMD EPYC 9254 24-Core Processor (4 threads; avx2 avx512f avx512_vnni)
- Memory: 7.7 GiB
- Disk: unknown
- System: Ubuntu 24.04.5 LTS, Linux 6.8.0-142-generic
- Compiler: gcc (Ubuntu 13.3.0-6ubuntu2~24.04.1) 13.3.0
- GPU seen by Vulkan: vulkaninfo not installed; Janas built without GPU support (glslangValidator not installed)
- Power: unknown
- Load when it started: 0.52 over 4 threads (the speed lines below say how idle the CPU was before each measurement)

### Results

```
quick answer fingerprint ok (c08f0f936eeba5cb)
quick speed measured with the CPU 98% idle before (load 0.95 over 4 threads; idle)
quick server 2/2 (chat, tool call)
```

### bench-quick

```
janas-bench unknown, 20261113-095316
system:  Linux 6.8.0-142-generic x86_64
CPU:     AMD EPYC 9254 24-Core Processor (4 performance cores, 4 threads on them, 4 usable in all)
CPU has: sse4.2 avx fma f16c avx2 avx512f avx512dq avx512bw avx512vl avx512vnni avx512-bf16 avx512vbmi avx512vbmi2 avx512ifma avx512vpopcntdq
kernels: avx512-vnni (512-bit tiles)
memory:  7.7 GiB, 3.8 GiB available; power: unknown
GPU:     none usable: turned off by JANAS_GPU=0
NPU:     none usable: turned off by JANAS_NPU=0
model:   Qwen3 4B Instruct Awq (qwen3), context 4096 tokens, expert cache 1.0 GiB, up to 4 compute threads, avx512-vnni (512-bit tiles) kernels, no GPU (turned off by JANAS_GPU=0), drafts from the conversation, down matrix at 4 bits (memory)
test:    prompt 256 tokens of prose (not repeated), 64 generated tokens, greedy

warming up the expert cache...
measuring 1 configurations, 3 rounds...

configuration               prefill       decode                               cache
4 threads                  69.6 tok/s    5.8 tok/s                               68.5%

result:  prefill 69.6 tok/s, decode 5.8 tok/s
         (the configuration the engine chose for each kind of pass, as it runs in a chat)
choices: mode max; one token: 4 threads (124.6 ms/token); 2-16 tokens: 4 threads; 17-63 tokens: 4 threads; blocks: 4 threads
```
