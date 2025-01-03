!pip install onnx
!pip install onnxruntime
!pip install torch
!pip install transformers
!pip install optimum
!pip install pyngrok
!ngrok authtoken 2kVnDlGDKJH5GwGwMhizZ7P3HkZ_5QXmUT2a74z32EBKWssJn
import time
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

from transformers import AutoTokenizer
from optimum.onnxruntime import ORTModelForCausalLM
import torch

class LlmManager:
  def __init__(self, hf_optimum_model_name):
    self.model = ORTModelForCausalLM.from_pretrained(hf_optimum_model_name)
    self.tokenizer = AutoTokenizer.from_pretrained(hf_optimum_model_name)

  def answerPrompt(self, input_text):
    inputs = self.tokenizer(input_text, return_tensors="pt")
    gen_tokens = self.model.generate(**inputs,do_sample=True,temperature=0.9, min_length=20,max_length=35)
    answer = self.tokenizer.batch_decode(gen_tokens)
    return answer

llm_man=LlmManager("optimum/gpt2")
llm_man.answerPrompt("Today is 30 celsius. I will take clothes: ")

@app.route("/clothesSuggest", methods=['POST', 'GET'])
def promptLLM():
  answer = ""
  if request.method=='GET':
    temp = request.values.get('temp')
    prompt = f"Today is {temp} celsius. I will take clothes: "
    print(prompt)
    answer = llm_man.answerPrompt(prompt)
  return answer


print('Prompt:')
t = time.process_time()
print(llm_man.answerPrompt("Today is 30 celsius. I will take clothes: "))
elapsed_time = time.process_time() - t
print(elapsed_time)

print("You can access the service publicly at:")
print(public_url)

threading.Thread(target=app.run, kwargs={"use_reloader": False}).start()
