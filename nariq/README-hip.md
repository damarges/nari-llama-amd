# Build and validate NariQ on gfx1100

Use a Linux ROCm 7.2.4 development environment with access to the AMD
GPU. In a container, pass through /dev/kfd and the appropriate /dev/dri
render device. Build from the patched NariQ tree.

```sh
CC=/opt/rocm/bin/amdclang CXX=/opt/rocm/bin/amdclang++ \
cmake -G Ninja -S . -B build-hip -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_HIP_COMPILER=/opt/rocm/lib/llvm/bin/clang \
  -DGGML_HIP=ON -DAMDGPU_TARGETS=gfx1100 \
  -DLLAMA_BUILD_TESTS=ON -DLLAMA_USE_PREBUILT_UI=OFF
cmake --build build-hip --parallel 6 --target llama-server test-trellis test-trellis-model
```

Obtain Qwen3.8-27B-NariQ-3.0bpw.gguf from the public Nari Labs model card
and record its SHA-256. Start the text-only server.

```sh
MODEL=/path/to/Qwen3.8-27B-NariQ-3.0bpw.gguf
build-hip/bin/llama-server -m "$MODEL" -ngl 99 --fit off \
  --host 127.0.0.1 --port 18281 -c 262144 -np 1 -fa on \
  -ctk q8_0 -ctv q8_0 -b 1024 -ub 1024
```

For the other cache runs, restart with q8_0/q4_0 and q4_0/q4_0. Submit
identical token IDs via /completion with cache_prompt=false, temperature=0
and n_predict=16. Record the actual prompt and generated token counts,
timings, and VRAM after completion. The companion benchmark script generates
the same prompt and prints those measurements.

To run the synthetic tensor test, use `build-hip/bin/test-trellis -v ROCm0`.

## Numerical parity procedure

Use the same coded weights for the comparison. Build the F32 reference with
nariq/decode_reference.py, passing the ggml-base shared library from this fork
and --outtype f32. Allow disk space for roughly 110 GB of reference weights
and use CPU inference for the F32 reference.

```sh
PROBE=build-hip/bin/test-trellis-model
"$PROBE" --model /path/to/reference-f32.gguf --output ref \
  --gpu-layers 0 --ubatch 8 --predict 8
for ubatch in 8 1; do
  "$PROBE" --model "$MODEL" --output "hip-$ubatch" --force ref.json \
    --gpu-layers 99 --ubatch "$ubatch" --predict 8
  python3 nariq/compare_logits.py ref "hip-$ubatch" \
    --max-kl 0.001 --min-top1 0.98 --output "hip-$ubatch-report.json"
done
```

This procedure uses the upstream comparison tool with max KL capped at 0.001. The larger
CUDA qualification samples 510 positions. Record mean KL, top-1
agreement, and greedy-token equality. Keep CUDA regression qualification
separate and validate it on CUDA hardware using the patched tree.

Upstream qualification record
https://github.com/nari-labs/nari-llama.cpp/blob/nariq/b12431/nariq/cuda-qualification.md

## Measured numerical results

The synthetic test failed at TRL1, mode 0, 35 input tokens, row 0
(HIP -10.3191, CPU -10.3261). The same-GGUF model probe used the CPU
backend with microbatch 8 and the same forced continuation for both
HIP runs. Each comparison covered 12 full-vocabulary logit rows.

| HIP microbatch | Mean KL | Max KL | Top-1 | Greedy tokens equal |
| --- | ---: | ---: | ---: | --- |
| 1 | 6.95e-9 | 1.99e-8 | 12/12 | Yes |
| 8 | 6.46e-9 | 1.81e-8 | 12/12 | Yes |

To reproduce the same-GGUF comparison, use the original GGUF in the
CPU reference command. The independent F32 artifact was generated,
but CPU inference on it was stopped under memory and disk pressure.
The table above uses the CPU backend of the coded GGUF.
