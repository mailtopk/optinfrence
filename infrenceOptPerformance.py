import tensorrt as trt
import pycuda.driver as cuda
import pycuda.autoinit
import numpy as np
import time

TRT_LOGGER = trt.Logger(trt.Logger.WARNING)
engine_file_path = './enginefiles/resnet50_fp16_engine_pytorch.plan'

#load engine/planer file
with open(engine_file_path, "rb") as f, trt.Runtime(TRT_LOGGER) as runtime:
    engine = runtime.deserialize_cuda_engine(f.read())

context = engine.create_execution_context()



# Get tensor names

input_name = None
output_name = None

for i in range(engine.num_io_tensors):
    name = engine.get_tensor_name(i)
    mode = engine.get_tensor_mode(name)
    if mode == trt.TensorIOMode.INPUT:
        input_name = name
    else:
        output_name = name

# Get shapes & dtypes from engine
input_shape = context.get_tensor_shape(input_name)
output_shape = context.get_tensor_shape(output_name)

input_dtype = trt.nptype(engine.get_tensor_dtype(input_name))
output_dtype = trt.nptype(engine.get_tensor_dtype(output_name))


#Allocate host memory
h_input = np.random.rand(*input_shape).astype(input_dtype)
h_output = np.empty(output_shape, dtype=output_dtype)

# Allocate device memory
d_input = cuda.mem_alloc(int(h_input.nbytes))
d_output = cuda.mem_alloc(int(h_output.nbytes))

# set tensor addresses 
context.set_tensor_address(input_name, int(d_input))
context.set_tensor_address(output_name, int(d_output))

stream = cuda.Stream()


#warm up 
for _ in range(10):
    cuda.memcpy_htod_async(d_input, h_input, stream)
    context.execute_async_v3(stream_handle=stream.handle)
    cuda.memcpy_dtoh_async(h_output, d_output, stream)
    stream.synchronize()
    
start_time = time.time()

for _ in range(1000):
    cuda.memcpy_htod_async(d_input, h_input, stream)
    context.execute_async_v3(stream_handle=stream.handle)
    cuda.memcpy_dtoh_async(h_output, d_output, stream)
    stream.synchronize()

end_time = time.time()

print("Avg latency : ", (end_time - start_time) / 100 * 1000, "ms")
