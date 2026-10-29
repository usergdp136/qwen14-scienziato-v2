# Merge fisico bf16 della scala scelta: base + lora/<name> (lora_B gia' scalata) -> /workspace/merged
import sys, torch, shutil, glob
from transformers import AutoModelForCausalLM
from peft import PeftModel
name = sys.argv[1]
m = AutoModelForCausalLM.from_pretrained('/workspace/base', dtype=torch.bfloat16, device_map={'': 0})
m = PeftModel.from_pretrained(m, '/workspace/lora/' + name); m = m.merge_and_unload(); m.save_pretrained('/workspace/merged', safe_serialization=True)
for f in glob.glob('/workspace/base/tokenizer*') + glob.glob('/workspace/base/vocab.json') + glob.glob('/workspace/base/merges.txt') + glob.glob('/workspace/base/generation_config.json'): shutil.copy(f, '/workspace/merged/')
print('merged', name)
