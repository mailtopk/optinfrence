import tensorrt as trt
import pycuda.driver as cudadriv
import pycuda.autoinit
import numpy as np
import time

# load Engine
TRT_LOGGER = trt.Logger(trt.Logger.WARNING)
engine_file_path = './enginefiles/resnet50_fp16_engine_pytorch.plan'

#load engine/planer file

with open(engine_file_path, "rb") as f :
    runtime = trt.Runtime(TRT_LOGGER)
    engine = runtime.deserialize_cuda_engine(f.read())
    context = engine.create_execution_context()
    
# Get tensor names
input_name = None
output_name = None

for i in range(engine.num_io_tensors):
    name = engine.get_tensor_name(i)
    mode = engine.get_tensor_mode(name)
    if mode == trt.TensorIOMode.INPUT:
        input_name = name # data
    else:
        output_name = name # resnetv17_dense0_fwd

# Get shapes & dtypes from engine
input_shape = context.get_tensor_shape(input_name)
output_shape = context.get_tensor_shape(output_name)

input_dtype = trt.nptype(engine.get_tensor_dtype(input_name))
output_dtype = trt.nptype(engine.get_tensor_dtype(output_name))


# multi-Stream Benchmarking
def benchmark_streams(engine, num_streams):
    # 1. Setup contexts and memory for each stream
    streams_data = []
    for i in range(num_streams):
        context = engine.create_execution_context()
        stream = cudadriv.Stream()
        
        # Allocate pinned host & device memory for this specific stream
        h_in = cudadriv.pagelocked_empty(tuple(input_shape), dtype=input_dtype)
        h_out = cudadriv.pagelocked_empty(tuple(output_shape), dtype=output_dtype)
        d_in = cudadriv.mem_alloc(h_in.nbytes)
        d_out = cudadriv.mem_alloc(h_out.nbytes)
        
        # Map addresses for this context
        context.set_tensor_address("data", int(d_in))
        context.set_tensor_address("resnetv17_dense0_fwd", int(d_out))
        
        streams_data.append({
            'ctx': context, 'st': stream, 
            'h_in': h_in, 'h_out': h_out, 
            'd_in': d_in, 'd_out': d_out
        })

    # 2. Timing Events
    start, end = cudadriv.Event(), cudadriv.Event()
    
    # 3. Execution (Simulate concurrent streams)
    start.record()
    for s in streams_data:
        # Launch H2D and Kernel asynchronously across different streams
        cudadriv.memcpy_htod_async(s['d_in'], s['h_in'], s['st'])
        s['ctx'].execute_async_v3(stream_handle=s['st'].handle)
        cudadriv.memcpy_dtoh_async(s['h_out'], s['d_out'], s['st'])
    
    # Wait for the slowest stream to finish
    for s in streams_data:
        s['st'].synchronize()
        
    end.record()
    end.synchronize()
    
    total_ms = end.time_since(start)
    return total_ms



# Run the test for 1 to 10 streams
print(f"{'Streams':<10} | {'Latency (ms)':<15} | {'Status'}")
print("-" * 40)
for n in range(1, 11):
    latency = benchmark_streams(engine, n)
    status = "OK" if latency < 33.3 else "OVER LIMIT"
    print(f"{n:<10} | {latency:<15.3f} | {status}")


