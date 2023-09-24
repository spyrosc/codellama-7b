import accelerate
import transformers
import json

model_id = "codellama/CodeLlama-7b-Instruct-hf"

config = transformers.AutoConfig.from_pretrained(model_id)

with accelerate.init_empty_weights():
    fake_model = transformers.AutoModelForCausalLM.from_config(config)

device_map = accelerate.infer_auto_device_map(fake_model, max_memory={0: "3GiB", "cpu": "24GiB"})
print(json.dumps(device_map, indent=4))
