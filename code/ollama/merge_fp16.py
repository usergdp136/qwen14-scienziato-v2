# Merge della LoRA nei pesi base con somma in fp32 e salvataggio in fp16 (NON bf16: il bf16 cancella la delta, vedi articolo §3.5).
# W = fp16( W0 + s * (alpha/r) * B A ). Uso: python merge_fp16.py <base_dir> <adapter_dir> <out_dir> [scala=1.0]
import json, sys, os, shutil, glob, torch
from safetensors import safe_open
from safetensors.torch import load_file, save_file
base, adp, out = sys.argv[1:4]; s = float(sys.argv[4]) if len(sys.argv) > 4 else 1.0
os.makedirs(out, exist_ok=True)
A = load_file(adp + '/adapter_model.safetensors'); cfg = json.load(open(adp + '/adapter_config.json')); sc = s * cfg['lora_alpha'] / cfg['r']
n = 0
for sh in sorted(glob.glob(base + '/*.safetensors')):
    t = {}
    with safe_open(sh, 'pt') as f:
        for name in f.keys():
            w = f.get_tensor(name); b = name[:-len('.weight')] if name.endswith('.weight') else name
            ka = 'base_model.model.' + b + '.lora_A.weight'; kb = 'base_model.model.' + b + '.lora_B.weight'
            if ka in A:
                w = (w.float() + sc * (A[kb].float() @ A[ka].float())); n += 1
            t[name] = w.to(torch.float16).contiguous()
    save_file(t, out + '/' + os.path.basename(sh), metadata={'format': 'pt'}); print(sh, len(t), flush=True)
for fn in glob.glob(base + '/*.json') + glob.glob(base + '/*.txt'): shutil.copy(fn, out)
c = json.load(open(out + '/config.json')); c['torch_dtype'] = 'float16'; c['dtype'] = 'float16'; json.dump(c, open(out + '/config.json', 'w'), indent=2)
print('matrici fuse:', n, '(attese 280)')
