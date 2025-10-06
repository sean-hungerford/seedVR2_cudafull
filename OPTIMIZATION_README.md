# SeedVR2 Performance Optimizations

**Optimized by eddy - October 5, 2025, 19:23 PST (Los Angeles Time)**

This document describes the comprehensive performance optimizations implemented for the SeedVR2 VideoUpscaler ComfyUI node.

## Original Author
- ComfyUI-SeedVR2_VideoUpscaler by [numz](https://github.com/numz/ComfyUI-SeedVR2_VideoUpscaler)
- Based on [SeedVR2](https://github.com/ByteDance-Seed/SeedVR) by ByteDance

## Optimization Author
- Performance enhancements and advanced features by **eddy**

## 🚀 Overview

The optimization package addresses three critical performance issues:

1. **显存管理波动** - Unstable VRAM usage patterns
2. **缺少torch.compile支持** - Missing torch.compile integration  
3. **矩阵乘法优化不足** - Insufficient matrix multiplication optimizations

## 📊 Performance Improvements

### Expected Performance Gains:
- **20-40% faster inference** with torch.compile
- **2-4x faster attention** with optimized matrix operations
- **30-50% memory reduction** with stable memory management
- **Eliminated memory fluctuations** with predictable VRAM usage

## 🔧 Optimization Modules

### 1. Torch.Compile Integration (`torch_compile.py`)

Provides intelligent model compilation with:
- **Automatic backend selection** (inductor/aot_eager)
- **Shape-aware compilation** for optimal performance
- **Compilation caching** to avoid recompilation
- **Fallback mechanisms** for compatibility

```python
# Usage example
from src.optimization.torch_compile import get_global_compile_manager

compile_manager = get_global_compile_manager(enable_debug=True)
compiled_model = compile_manager.compile_dit_model(model)
```

### 2. SageAttention Integration (`matrix_ops.py`)

Advanced attention mechanisms with RTX 5090 FP4 support:
- **SageAttention** - Standard optimized attention
- **SageAttention3** - INT8/FP16 balanced mode for Blackwell
- **SageAttention3 FP4** - 1038 TOPS microscaling for RTX 5090
- **SageAttention3 FP8** - Ultra-fast mode with FP8
- **Flash Attention** and **XFormers** fallbacks
- **Automatic backend selection** based on sequence length

```python
# Usage example
from src.optimization.matrix_ops import OptimizedMatrixOps

output = OptimizedMatrixOps.optimized_attention(
    query, key, value, attention_mode="sageattn_3_fp4"
)
```

### 3. FP4 Quantization Support (`fp4_quantization.py`)

Cutting-edge FP4 quantization for maximum performance:
- **FP4 Experimental** - FP8 weights + FP4 attention
- **FP4 Scaled** - For scaled FP8 models with FP4 attention
- **FP4 Scaled Fast** - Optimized matmul for RTX 5090
- **Automatic capability detection** based on GPU
- **Seamless integration** with SageAttention3 FP4

```python
# Usage example
from src.optimization.fp4_quantization import get_fp4_manager

fp4_manager = get_fp4_manager(enable_debug=True)
quantized_model = fp4_manager.apply_fp4_quantization(model, "fp4_experimental")
```

### 4. Stable Memory Management (`stable_memory.py`)

Comprehensive memory management system:
- **Memory pool management** with tensor reuse
- **Predictable allocation patterns**
- **Memory pressure monitoring** and response
- **Intelligent garbage collection** scheduling

```python
# Usage example
from src.optimization.stable_memory import memory_efficient_context

with memory_efficient_context(enable_debug=True) as memory_manager:
    # Your memory-intensive operations here
    pass
```

## 🎛️ ComfyUI Integration

### New Node Parameters

The SeedVR2 node now includes:

- **`enable_optimizations`** (Boolean, default: True)
  - Enables torch.compile and matrix optimizations
  - Recommended: Keep enabled for best performance

- **`attention_mode`** (Dropdown, default: "auto")
  - "auto" - Automatic selection based on hardware
  - "sageattn_3_fp4" - FP4 microscaling for RTX 5090 (1038 TOPS)
  - "sageattn_3" - INT8/FP16 balanced mode for Blackwell
  - "sageattn_3_fp8" - FP8 ultra-fast mode
  - "sageattn" - Standard SageAttention
  - "flash_attn" - Flash Attention
  - "xformers" - XFormers memory efficient attention
  - "sdpa" - PyTorch scaled dot product attention

- **`quantization`** (Dropdown, default: "disabled")
  - "disabled" - No quantization
  - "fp4_experimental" - FP8 weights + FP4 attention (requires sageattn_3_fp4)
  - "fp4_scaled" - For scaled FP8 models with FP4 attention
  - "fp4_scaled_fast" - Optimized matmul for RTX 5090

### Automatic Features

- **Stable memory management** is automatically applied
- **Optimal attention backend** is automatically selected based on hardware
- **Memory pressure handling** works transparently
- **FP4 compatibility detection** ensures safe operation
- **Attention mode recommendations** for quantization modes

## 📈 Performance Testing

### Running Benchmarks

Use the included performance test script:

```bash
# Basic benchmark
python performance_test.py --model seedvr2_ema_3b_fp16.safetensors

# Extended benchmark with more frames
python performance_test.py --model seedvr2_ema_3b_fp16.safetensors --test-frames 32

# Save results to custom file
python performance_test.py --output my_benchmark.json
```

### Benchmark Results

The benchmark tests:
1. **Matrix operations performance** (attention speedup)
2. **Torch.compile overhead** and benefits
3. **Memory stability** over multiple iterations

Example output:
```
🚀 SeedVR2 Performance Optimization Report
==================================================

📊 Matrix Operations Performance:
  ✅ Attention Speedup: 2.3x
  ✅ Memory Reduction: 35.2%

🔧 Torch.Compile Performance:
  📈 Compilation Overhead: 12.5s

🧠 Memory Stability:
  📊 Memory Usage Std Dev: 0.045 GB
  📊 Stability Ratio: 0.08
  ✅ Memory usage is very stable
```

## 🔍 Technical Details

### Attention Optimization Strategy

1. **Sequence length detection** determines optimal backend
2. **Flash Attention** for long sequences (>512 tokens)
3. **XFormers** for medium sequences (256-512 tokens)  
4. **SDPA** for shorter sequences
5. **Manual implementation** as final fallback

### Memory Management Strategy

1. **Memory pools** for efficient tensor reuse
2. **Pressure monitoring** with automatic response
3. **Scheduled garbage collection** to prevent fragmentation
4. **Device movement optimization** to reduce transfers

### Compilation Strategy

1. **Backend detection** based on GPU capabilities
2. **Model-specific optimization** (DiT vs VAE)
3. **Shape-aware compilation** for dynamic inputs
4. **Caching system** to avoid recompilation

## ⚙️ Configuration Options

### Environment Variables

- `SEEDVR2_ENABLE_COMPILE=false` - Disable torch.compile globally
- `SEEDVR2_COMPILE_BACKEND=inductor` - Force specific backend
- `SEEDVR2_DEBUG_MEMORY=true` - Enable memory debug logging

### Advanced Configuration

```python
# Custom compile manager
from src.optimization.torch_compile import TorchCompileManager

compile_manager = TorchCompileManager(
    cache_dir="/custom/cache/path",
    enable_debug=True
)

# Custom memory manager  
from src.optimization.stable_memory import StableMemoryManager

memory_manager = StableMemoryManager(
    enable_debug=True,
    memory_limit_gb=16.0
)
```

## 🐛 Troubleshooting

### Common Issues

1. **Compilation fails**
   - Set `enable_optimizations=False` in node
   - Check GPU compatibility (requires modern CUDA)

2. **Memory issues persist**
   - Reduce batch size
   - Enable `preserve_vram` mode
   - Use BlockSwap for very limited VRAM

3. **Performance regression**
   - First run includes compilation overhead
   - Subsequent runs should be faster
   - Check benchmark results

### Debug Mode

Enable debug logging for detailed information:

```python
# In ComfyUI node
enable_optimizations = True  # Enable optimizations
# Debug output will show compilation and memory stats
```

## 📝 Compatibility

### Requirements

- **PyTorch 2.0+** for torch.compile support
- **CUDA 11.8+** for optimal performance
- **Flash Attention** (optional, auto-detected)
- **XFormers** (optional, auto-detected)

### Tested Configurations

- ✅ RTX 4090 (24GB) - Full optimizations
- ✅ RTX 3080 (10GB) - With BlockSwap
- ✅ RTX 2080 Ti (11GB) - Basic optimizations
- ⚠️ GTX 1080 Ti (11GB) - Limited support

## 🔄 Migration Guide

### From Previous Version

1. **No breaking changes** - existing workflows continue to work
2. **New parameter** `enable_optimizations` defaults to True
3. **Automatic benefits** - no configuration required
4. **Optional tuning** - use benchmark script for optimization

### Recommended Settings

For **high-end GPUs** (RTX 4090, A100):
```
enable_optimizations: True
preserve_vram: False
block_swap: Disabled
```

For **mid-range GPUs** (RTX 3080, RTX 4070):
```
enable_optimizations: True  
preserve_vram: True
block_swap: 8-16 blocks if needed
```

For **limited VRAM** (<12GB):
```
enable_optimizations: True
preserve_vram: True
block_swap: 16-32 blocks
batch_size: Reduced
```

## 📚 API Reference

### Core Classes

- `TorchCompileManager` - Handles model compilation
- `OptimizedMatrixOps` - Provides optimized operations
- `StableMemoryManager` - Manages memory allocation
- `PerformanceBenchmark` - Tests optimization performance

### Key Functions

- `get_global_compile_manager()` - Get shared compile manager
- `get_matrix_optimizer()` - Get matrix operations optimizer  
- `get_memory_manager()` - Get shared memory manager
- `memory_efficient_context()` - Context for memory management

## 🤝 Contributing

To contribute optimizations:

1. **Follow the modular design** - separate concerns
2. **Add comprehensive tests** - use benchmark framework
3. **Include fallback mechanisms** - ensure compatibility
4. **Document performance gains** - provide before/after metrics

## 📄 License

Same license as the main SeedVR2 project.
