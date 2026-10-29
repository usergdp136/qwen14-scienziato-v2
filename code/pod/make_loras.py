# Scala lora_B di s (= scala la delta alpha/r*B@A di s) per 0.3/0.5/0.7/0.85/1.0 e scrive target_modules come lista.
import json, sys, pathlib
from safetensors.torch import load_file, save_file
src = pathlib.Path('/workspace/adapter'); cfg = json.load(open(src / 'adapter_config.json')); cfg['target_modules'] = ['q_proj', 'k_proj', 'v_proj', 'o_proj', 'gate_proj', 'up_proj', 'down_proj']
Wt = load_file(str(src / 'adapter_model.safetensors'))
for name, s in [('ft', 1.0), ('s085', 0.85), ('s07', 0.7), ('s05', 0.5), ('s03', 0.3)]:
    d = pathlib.Path('/workspace/lora') / name; d.mkdir(parents=True, exist_ok=True)
    save_file({k: (v * s if 'lora_B' in k else v).contiguous() for k, v in Wt.items()}, str(d / 'adapter_model.safetensors'), metadata={'format': 'pt'})
    json.dump(cfg, open(d / 'adapter_config.json', 'w'), indent=1); print(name, s, len(Wt))
