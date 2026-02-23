#include <iostream>
#include <fstream>
#include <vector>
#include <memory>
#include <cuda_runtime_api.h>
#include <NvInfer.h>
#include <nvtx3/nvToolsExt.h>

using namespace nvinfer1;



//$ nvcc benchmark_reset.cu -o benchmark -lnvinfer -lnvToolsExt -I/usr/include/x86_64-linux-gun -L/usr/lib/x86_64-linux-gnu

// Logger for TensorRT
class Logger : public ILogger {
    void log(Severity severity, const char* msg) noexcept override {
        if (severity <= Severity::kWARNING) std::cout << msg << std::endl;
    }
} gLogger;

// Structure to hold resources for each stream
struct StreamResources {
    std::unique_ptr<IExecutionContext> context;
    cudaStream_t stream;
    void* d_in;
    void* d_out;
    float* h_in;
    float* h_out;
    size_t in_size;
    size_t out_size;
};

float benchmark_streams(ICudaEngine* engine, int num_streams, const char* input_name, const char* output_name) {
    std::vector<StreamResources> resources(num_streams);
    
    // 1. Setup contexts and memory for each stream
    for (int i = 0; i < num_streams; ++i) {
        resources[i].context = std::unique_ptr<IExecutionContext>(engine->createExecutionContext());
        cudaStreamCreate(&resources[i].stream);

        auto in_dims = engine->getTensorShape(input_name);
        auto out_dims = engine->getTensorShape(output_name);
        
        resources[i].in_size = 1;
        for (int j = 0; j < in_dims.nbDims; ++j) 
        	resources[i].in_size *= in_dims.d[j];
        	resources[i].out_size = 1;
        for (int j = 0; j < out_dims.nbDims; ++j) 
        	resources[i].out_size *= out_dims.d[j];

        // Allocate Pinned Host Memory (equivalent to pagelocked_empty)
        cudaMallocHost(&resources[i].h_in, resources[i].in_size * sizeof(float));
        cudaMallocHost(&resources[i].h_out, resources[i].out_size * sizeof(float));
        
        // Allocate Device Memory
        cudaMalloc(&resources[i].d_in, resources[i].in_size * sizeof(float));
        cudaMalloc(&resources[i].d_out, resources[i].out_size * sizeof(float));

        // Set Addresses
        resources[i].context->setTensorAddress(input_name, resources[i].d_in);
        resources[i].context->setTensorAddress(output_name, resources[i].d_out);
    }

    // 2. Timing Events
    cudaEvent_t start, end;
    cudaEventCreate(&start);
    cudaEventCreate(&end);

    cudaEventRecord(start);
    nvtxRangePushA(("Batch_of_" + std::to_string(num_streams)).c_str());

    // 3. Execution (Asynchronous Launch)
    for (int i = 0; i < num_streams; ++i) {
        nvtxRangePushA(("Launch_Stream_" + std::to_string(i)).c_str());
        
        cudaMemcpyAsync(resources[i].d_in, resources[i].h_in, resources[i].in_size * sizeof(float), cudaMemcpyHostToDevice, resources[i].stream);
        resources[i].context->enqueueV3(resources[i].stream);
        cudaMemcpyAsync(resources[i].h_out, resources[i].d_out, resources[i].out_size * sizeof(float), cudaMemcpyDeviceToHost, resources[i].stream);
        
        nvtxRangePop();
    }

    nvtxRangePop();

    // 4. Synchronize all streams
    for (int i = 0; i < num_streams; ++i) {
        cudaStreamSynchronize(resources[i].stream);
    }

    cudaEventRecord(end);
    cudaEventSynchronize(end);

    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, end);

    // Cleanup
    for (int i = 0; i < num_streams; ++i) {
        cudaFree(resources[i].d_in);
        cudaFree(resources[i].d_out);
        cudaFreeHost(resources[i].h_in);
        cudaFreeHost(resources[i].h_out);
        cudaStreamDestroy(resources[i].stream);
    }
    cudaEventDestroy(start);
    cudaEventDestroy(end);

    return milliseconds;
}

int main() {

    std::string plan_path = "../enginefiles/resnet50_fp16_engine_pytorch.plan";
    std::cout<<"file Path is : " << plan_path << std::endl;
    
    std::ifstream file(plan_path, std::ios::binary);
    if (!file.good()) 
    	return -1;

    file.seekg(0, file.end);
    size_t size = file.tellg();
    file.seekg(0, file.beg);
    std::vector<char> engine_data(size);
    file.read(engine_data.data(), size);

    std::unique_ptr<IRuntime> runtime{createInferRuntime(gLogger)};
    std::unique_ptr<ICudaEngine> engine{runtime->deserializeCudaEngine(engine_data.data(), size)};

    const char* input_name = engine->getIOTensorName(0);
    const char* output_name = engine->getIOTensorName(1);

    printf("%-10s | %-15s | %-s\n", "Streams", "Latency (ms)", "Status");
    printf("----------------------------------------\n");

    for (int n = 1; n <= 10; ++n) {
        float latency = benchmark_streams(engine.get(), n, input_name, output_name);
        const char* status = (latency < 33.3) ? "OK" : "OVER LIMIT";
        printf("%-10d | %-15.3f | %-s\n", n, latency, status);
    }

    return 0;
}

