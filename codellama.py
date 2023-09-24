from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
import torch

model_id = "codellama/CodeLlama-7b-Instruct-hf"
quantization_config = BitsAndBytesConfig(
   load_in_4bit=True,
   bnb_4bit_compute_dtype=torch.float16,
   llm_int8_enable_fp32_cpu_offload=True
)

tokenizer = AutoTokenizer.from_pretrained(model_id)

device_map = {
    "model.embed_tokens": 0,
    "model.layers.0": 0,
    "model.layers.1": 0,
    "model.layers.2": 0,
    "model.layers.3": 0,
    "model.layers.4": 0,
    "model.layers.5": 0,
    "model.layers.6": 0,
    "model.layers.7": 0,
    "model.layers.8": 0,
    "model.layers.9": 0,
    "model.layers.10": 0,
    "model.layers.11": 0,
    "model.layers.12": 0,
    "model.layers.13": 0,
    "model.layers.14": 0,
    "model.layers.15": 0,
    "model.layers.16": 0,
    "model.layers.17": 0,
    "model.layers.18": 0,
    "model.layers.19": 0,
    "model.layers.20": "cpu",
    "model.layers.21": "cpu",
    "model.layers.22": "cpu",
    "model.layers.23": "cpu",
    "model.layers.24": "cpu",
    "model.layers.25": "cpu",
    "model.layers.26": "cpu",
    "model.layers.27": "cpu",
    "model.layers.28": "cpu",
    "model.layers.29": "cpu",
    "model.layers.30": "cpu",
    "model.layers.31": "cpu",
    "model.norm": 0,
    "lm_head": 0 
}
 
 

model = AutoModelForCausalLM.from_pretrained(
    model_id,
    quantization_config=quantization_config,
    device_map=device_map,
)

prompt = 'Write a fibonacci function in go'
inputs = tokenizer(prompt, return_tensors="pt").to("cuda")

output = model.generate(
    inputs["input_ids"],
    max_new_tokens=200,
    do_sample=True,
    top_p=0.9,
    temperature=0.1,
)
output = output[0].to("cpu")
print(tokenizer.decode(output))
