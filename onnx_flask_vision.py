!pip install pyngrok
!pip install onnxruntime


!ngrok authtoken YOUR_NGROK_TOKEN
import os
import threading
os.environ["FLASK_DEBUG"]="1"
from flask import Flask
from pyngrok import ngrok
from flask import request
app = Flask(__name__)
port = "5000"

public_url = ngrok.connect(port).public_url

app.config["BASE_URL"] = public_url

public_url = ngrok.connect(port).public_url

import torch

import onnxruntime
import numpy as np
from PIL import Image
import torchvision.transforms as transforms


class Utilities:
# Preprocess the input image
  def preprocess_image_alexnet(self, image_path):
      transform = transforms.Compose([
          transforms.Resize((224, 224)),
          transforms.ToTensor(),
          transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
      ])
      image = Image.open(image_path).convert("RGB")
      tensor = transform(image).unsqueeze(0)  # Add batch dimension
      return tensor.numpy()

  def map_classes_alexnet(self, class_number):
    imagenet_classes = []
    with open("imagenet_classes.txt", "r") as f:
      imagenet_classes = [line.strip() for line in f]
    # Map the number to the class name
    class_label = imagenet_classes[class_number]
    return class_label

ut = Utilities()


class ImageClassifierManager:
  def __init__(self, onnx_model_path):
    self.onnx_model_path = onnx_model_path
    self.session = onnxruntime.InferenceSession(self.onnx_model_path)

  def classifyImage(self, image_path):
    input_tensor = ut.preprocess_image_alexnet(image_path)

    # Get input and output names
    input_name = self.session.get_inputs()[0].name
    output_name = self.session.get_outputs()[0].name

    # Run inference
    outputs = self.session.run([output_name], {input_name: input_tensor})
    predicted_class = np.argmax(outputs[0])
    return ut.map_classes_alexnet(predicted_class)

im = ImageClassifierManager("alexnet_Opset16.onnx")
#image_path1 = "example1.jpg"

#print(im.classifyImage(image_path=image_path1))

@app.route("/classifyImage", methods=['POST', 'GET'])
def classifyImage():
  file = request.files['file']
  file.save("current.jpg")
  return im.classifyImage("current.jpg")

print("You can access the service publicly at:")
print(public_url)
threading.Thread(target=app.run, kwargs={"use_reloader": False}).start()