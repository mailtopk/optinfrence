import tensorrt as trt
import torch
from torchvision import transforms
from PIL import Image
import requests
from io import BytesIO
import json
from torchvision.models import ResNet50_Weights
import torchvision.models as models
import os


# Compare ResNet-50 TensorRT engine results with un-quantized model ResNet-50 
# Vanilla model file size 102 mb, Engine file size 51 mb

model_download_path = "../models"
os.environ['TORCH_HOME'] = model_download_path
pt_model = models.resnet50(weights='IMAGENET1K_V1').cuda().eval()


# ImageNet Preprocessing
preprocess = transforms.Compose(
    [
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ]
)

url='https://images.dog.ceo/breeds/retriever-golden/n02099601_3004.jpg'
response = requests.get(url)
response.raise_for_status()

image_data = BytesIO(response.content)
img = Image.open(image_data).convert('RGB')
input_tensor = preprocess(img).unsqueeze(0).cuda()

# Resnet vanilla infrence
with torch.no_grad():
    pt_output = pt_model(input_tensor)

pt_probs = torch.nn.functional.softmax(pt_output, dim=1)
pt_conf, pt_idx = torch.max(pt_probs, 1)


# TensorRT code

TRT_LOGGER = trt.Logger(trt.Logger.WARNING)
with open("../enginefiles/resnet50fp16.engine", 'rb') as f:
    runtime = trt.Runtime(TRT_LOGGER)
    engine = runtime.deserialize_cuda_engine(f.read())
    context = engine.create_execution_context()

# Engine file infrence
output_tensor = torch.empty((1, 1000), device='cuda')

input_name = engine.get_tensor_name(0)
output_name = engine.get_tensor_name(1)

context.set_input_shape(input_name, input_tensor.shape)
context.set_tensor_address(input_name, input_tensor.data_ptr())
context.set_tensor_address(output_name, output_tensor.data_ptr())

stream = torch.cuda.current_stream()
success = context.execute_async_v3(stream_handle=stream.cuda_stream)
stream.synchronize()

prob = torch.nn.functional.softmax(output_tensor, dim=1)
conf, idx = torch.max(prob, 1)

print('-'*50)
if success:
    print("Inference complete....")

    
    weights = ResNet50_Weights.DEFAULT
    categories = weights.meta['categories']

    print(f"Vanilla Model infrence Predicated Name {categories[pt_idx]}. Confidence : {pt_conf.item()*100: 2f} ")
    print(f'Predicted Name : {categories[idx.item()]}. Confidence : {conf.item()*100: 2f}')
else:
    print("Failed to run infrence code")
print('-'*50)

# Output
#--------------------------------------------------
#Inference complete....
#Vanilla Model infrence Predicated Name golden retriever. Confidence :  90.125465
#Predicted Name : golden retriever. Confidence :  89.988774
#--------------------------------------------------
