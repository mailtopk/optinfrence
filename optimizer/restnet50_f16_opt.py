import tensorrt as trt
import sys
import os

TRT_LOGGER = trt.Logger(trt.Logger.INFO)

def build_engine(onnx_file_path, engin_file_path):
    builder = trt.Builder(TRT_LOGGER)
    network = builder.create_network(
        1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
    )

    onnx_parser = trt.OnnxParser(network, TRT_LOGGER)

    with open(onnx_file_path, 'rb') as nx_model:
        if not onnx_parser.parse(nx_model.read()):
            print(f"Failed to parse onnx file {onnx_file_path}..")
            return None
    

    profile = builder.create_optimization_profile()
    input_tensor = network.get_input(0)
    input_name = input_tensor.name

    profile.set_shape(
        input_name,
        min=(1, 3, 224, 224),
        opt=(8, 3, 224, 224),
        max=(32, 3, 224, 224)
    )

    print("Building config and profile...")
    config = builder.create_builder_config()
    

    # config
    config.add_optimization_profile(profile)
    config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 1 << 30 ) #1GP
    #enable FP16 mode
    if builder.platform_has_fast_fp16:
        config.set_flag(trt.BuilderFlag.FP16)
    else:
        print("Failed - FP16 not supported on this GPU")
        return None

    
    # building Engine
    serialized_engine = builder.build_serialized_network(network, config)

    if serialized_engine is None:
        print("Failed to build engine")
        return None

    #save engine file
    with open(engin_file_path, 'wb') as f:
        f.write(serialized_engine)

    print(f"Completed optmization FP16. Engine file stored {engin_file_path}")

if __name__ == "__main__":
    onnx_model_path = "../models/resnet50-v1-7.onnx"
    enggine_output_path = "../enginefiles/resnet50fp16.engine"

    if not os.path.exists(onnx_model_path):
        print(f"model file not found at {onnx_model_path}")
        sys.exit(1)

    build_engine(onnx_model_path, enggine_output_path)