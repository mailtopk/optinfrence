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

# Convert trt.Dims to tuple for PyCUDA compatibility
host_input = cudadriv.pagelocked_empty(tuple(input_shape), dtype=input_dtype)
host_output = cudadriv.pagelocked_empty(tuple(output_shape), dtype=output_dtype)

# pre-fill input (if needed)
np.copyto(host_input, np.random.rand(*input_shape).astype(input_dtype))

# allocate device memory
device_input = cudadriv.mem_alloc(host_input.nbytes)
device_output = cudadriv.mem_alloc(host_output.nbytes)



# set tensor addresses 
context.set_tensor_address(input_name, int(device_input))
context.set_tensor_address(output_name, int(device_output))

# execution step
stream = cudadriv.Stream()


# create Cuda events for timing
start_event = cudadriv.Event()
end_event = cudadriv.Event()

# latencies stroage
iterations = 1000
latencies = []

print("warming up GPU..")
for _ in range(10):
	cudadriv.memcpy_htod_async(device_input, host_input, stream)
	context.execute_async_v3(stream_handle=stream.handle)
	cudadriv.memcpy_dtoh_async(host_output, device_output, stream)
stream.synchronize()

print(f"Running {iterations} ....")

for i in range(iterations):
	start_event.record()
	
	cudadriv.memcpy_htod_async(device_input, host_input, stream)
	context.execute_async_v3(stream_handle=stream.handle)
	cudadriv.memcpy_dtoh_async(host_output, device_output, stream)
	
	end_event.record()
	end_event.synchronize()
	
	# store in mil sec
	latency = end_event.time_since(start_event)
	latencies.append(latency)
	
# calculate
latencis = np.array(latencies)
avg_latency = np.mean(latencies)
p99_latency = np.percentile(latencis, 99)
p50_latency = np.percentile(latencis, 50)
fps = 1000.0/avg_latency

print("-" * 30)
print(f"Average Latency : {avg_latency:.3f} ms")
print(f"Media (P50) : {p50_latency:.3f} ms")
print(f"99th Percentile: {p99_latency:.3f} ms")
print(f"Throughtput: {fps:.2f} FPS")
print("-" * 30)
print()
