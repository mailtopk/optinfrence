# Inference Optimization & Memory Management
Target Hardware: NVIDIA Orin Nano (8GB) | JetPack 6.x | CUDA 12.6

1. TensorRT : Is designed for execution. Layer Fusion: TensorRT combines multiple layers (like Convolution, Bias, and ReLU) into a single "kernel" to reduce the number of times data is moved in/out of the GPU memory.
Precision Calibration: It can convert 32-bit (FP32) models to 16-bit (FP16) or 8-bit (INT8) using the Orin's specialized Tensor Cores, often doubling performance with negligible accuracy loss.
Kernel Auto-Tuning: During the "build" phase, TensorRT tests hundreds of different ways to run your specific model on the Orin Nano's 1024 cores and picks the mathematically fastest version for your specific hardware.
This code example is of FP16-bit and reset50 base model

3. Understanding the Memory "Handshake": In a standard PC, the CPU and GPU have separate RAM. On the Jetson Orin Nano, they share the same physical memory (Unified Memory), but they "see" it differently. To run inference, data must travel a specific path.
The Lifecycle of an Inference Frame:
- Host Allocation (CPU): Create a buffer in System RAM.
- Device Allocation (GPU): Reserve a specific address space in the GPU’s VRAM.
- H2D (Host to Device): Copy your image data from the CPU's space to the GPU's space.
- Inference: The GPU processes the data at that address.
- D2H (Device to Host): Copy the results (probabilities/boxes) back to the CPU's space so Python can read them.
- Optimization: Pinned (Pagelocked) Memory
Standard Python memory is "pageable," meaning the OS can move it around. This forces the CPU to do an extra internal copy before the GPU can see it.
We use cuda.pagelocked_empty to "pin" the memory. This tells the OS: "Don't move this; the GPU is going to talk to this address directly." ( only for Jetson nano orin )

5. Essential Memory Code Patterns : 
PyCUDA to handle these transfers reliably on Jetson.
Step A: The Setup (Once per App)
```python
# Create Pinned Host memory (CPU)
h_input = cuda.pagelocked_empty(tuple(input_shape), dtype=np.float32)
h_output = cuda.pagelocked_empty(tuple(output_shape), dtype=np.float32)

# Reserve Device memory (GPU)
d_input = cuda.mem_alloc(h_input.nbytes)
d_output = cuda.mem_alloc(h_output.nbytes)
```

Step B: The Transfer (Every Frame)
```python
# 1. Copy Image to GPU (Host to Device)
cuda.memcpy_htod_async(d_input, h_input, stream)

# 2. Run the Model
context.execute_async_v3(stream_handle=stream.handle)

# 3. Copy Results to CPU (Device to Host)
# Without this, 'h_output' will stay empty/zeros!
cuda.memcpy_dtoh_async(h_output, d_output, stream)

# 4. Wait for the GPU to finish
stream.synchronize()
```

Benchmarks (ResNet-style Model in MAXN Mode)
Configuration	Latency	Performance
1 Stream	2.39 ms	Ultra-Low Latency
10 Streams	27.59 ms	High-Density (30 FPS x 10)
5. Summary for New Developers

Allocate Once: (Importent) Never call mem_alloc or pagelocked_empty in your main loop. It is slow and causes memory leaks.
Sync is Key: Always stream.synchronize() before you try to read the values in h_output, otherwise the CPU might read the data before the GPU is finished writing it.
MAXN Mode: Always run sudo jetson_clocks before benchmarking to ensure the hardware isn't "sleeping" between frames.
