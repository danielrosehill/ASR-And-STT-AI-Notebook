# Model Format Guide - Whisper Fine-Tunes

When you fine-tune Whisper using a standard notebook example, you'll end up with a folder structure like this:

![alt text](screenshots/1.png)

The directory contains `runs` for resuming from checkpoints, but the core model file is:

**`model.safetensors`**

This file is directly usable for STT/inference. However, given the wide variety of Whisper deployment scenarios and the fact that you may want to use your fine-tuned model on devices with different processing capabilities, you'll likely need to convert it to other formats.

## Common Model Formats

### SafeTensors (Original Format)

- **File extension**: `.safetensors`
- **Use case**: Direct inference, training, PyTorch-based applications
- **Advantages**: Safe serialization format, prevents arbitrary code execution
- **Disadvantages**: Limited compatibility with optimized inference engines

### GGML (Legacy)

- **File extension**: `.bin`
- **Use case**: CPU-only inference on edge devices (e.g., FUTO Keyboard)
- **Compatible with**: `whisper.cpp` (older versions)
- **Advantages**: Enables deployment on resource-constrained devices, edge computing
- **Status**: Legacy format, superseded by GGUF
- **Considerations**: Hardware limitations still apply. Consider converting Tiny models to GGML for mobile/edge deployment while keeping Base or larger models in CTranslate2 for desktop applications.

### GGUF (Recommended for Edge/CPU)

- **File extension**: `.gguf`
- **Use case**: CPU-only inference on edge devices, local deployment
- **Compatible with**: Modern `whisper.cpp`, llama.cpp ecosystem
- **Advantages**:
  - Improved metadata handling (embedded model info, hyperparameters)
  - Better version control and compatibility checking
  - More efficient storage and loading
  - Standardized format across llama.cpp ecosystem
  - Supports quantization for smaller model sizes
- **Best for**: Modern edge deployments, CPU inference, resource-constrained environments
- **Migration**: GGUF is the successor to GGML and should be preferred for new projects

### CTranslate2

- **File extension**: `.bin` (directory with model files)
- **Use case**: Optimized inference for desktop applications
- **Compatible with**: Faster Whisper and many local STT applications
- **Advantages**: Significantly faster inference, reduced memory usage, optimized for CPU and GPU
- **Best for**: Production deployments requiring speed and efficiency

### ONNX

- **File extension**: `.onnx`
- **Use case**: Cross-platform deployment, inference optimization
- **Compatible with**: ONNX Runtime, various inference engines
- **Advantages**: Hardware-agnostic, works across different ML frameworks. Long recording durations / less chunking. 
- **Best for**: Applications requiring maximum portability across platforms and hardware

### Core ML (Apple Devices)

- **File extension**: `.mlmodel` or `.mlpackage`
- **Use case**: iOS, macOS, and Apple Silicon deployment
- **Advantages**: Native Apple Neural Engine acceleration, optimized battery usage
- **Best for**: Native Apple applications

### TensorFlow Lite

- **File extension**: `.tflite`
- **Use case**: Mobile deployment (Android/iOS)
- **Advantages**: Lightweight, optimized for mobile inference
- **Best for**: Mobile applications with size and performance constraints

## Format Selection Guide

| Format | Best Use Case | Performance | Compatibility |
|--------|---------------|-------------|---------------|
| SafeTensors | Training, PyTorch apps | Baseline | PyTorch ecosystem |
| GGML | Legacy edge devices | Optimized for CPU | Old whisper.cpp projects |
| GGUF | Modern edge devices, CPU-only | Optimized for CPU | Modern whisper.cpp, llama.cpp |
| CTranslate2 | Desktop apps, servers | High (GPU/CPU) | Faster Whisper, production apps |
| ONNX | Cross-platform deployment | Good | Wide framework support |
| Core ML | Apple devices | Excellent (on Apple HW) | Apple ecosystem only |
| TFLite | Mobile apps | Good | Android/iOS |

## Conversion Considerations

- **Model size**: Larger models (Base, Small, Medium) may not be practical for GGML/GGUF on edge devices
- **Target hardware**: GPU availability, CPU capabilities, RAM constraints
- **Use case**: Real-time vs. batch processing, latency requirements
- **Deployment environment**: Cloud, edge, mobile, desktop