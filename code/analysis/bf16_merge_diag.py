# Diagnostica post-hoc (non pre-registrata): quanto della delta LoRA sopravvive all'arrotondamento a bf16 quando si fa il merge fisico W = bf16(W0 + dW)?
# Uso: python bf16_merge_diag.py <shard_base.safetensors> <adapter_model.safetensors> [out.json]
import json, sys, torch, statistics as st
from safetensors import safe_open
from safetensors.torch import load_file
shard, adapter = sys.argv[1], sys.argv[2]; out = sys.argv[3] if len(sys.argv) > 3 else 'bf16_merge_diag.json'
A = load_file(adapter); cfg = json.load(open(adapter.replace('adapter_model.safetensors', 'adapter_config.json'))); sc = cfg['lora_alpha'] / cfg['r']
res = []
with safe_open(shard, 'pt') as f:
    for name in f.keys():
        if not name.endswith('.weight') or '.layers.' not in name: continue
        base = name[:-len('.weight')]; ka = 'base_model.model.' + base + '.lora_A.weight'; kb = 'base_model.model.' + base + '.lora_B.weight'
        if ka not in A: continue
        W0 = f.get_tensor(name)                       # bf16
        dW = sc * (A[kb].float() @ A[ka].float())      # fp32, (alpha/r) B A
        Wm = (W0.float() + dW).to(torch.bfloat16)      # merge fisico in bf16
        eff = Wm.float() - W0.float()                  # delta effettivamente rimasta
        nd = dW.norm().item(); err = (eff - dW).norm().item()
        cos = torch.nn.functional.cosine_similarity(eff.flatten(), dW.flatten(), dim=0).item()
        res.append({'name': base, 'rel_norm_dW_over_W0': nd / W0.float().norm().item(), 'rel_err': err / nd, 'cos': cos,
                    'frac_elems_unchanged': (eff == 0).float().mean().item(), 'frac_dW_below_half_ulp': ((dW.abs() < (W0.float().abs() * 2 ** -9))).float().mean().item()})
        print(base, round(res[-1]['rel_err'], 3), round(cos, 3), round(res[-1]['frac_elems_unchanged'], 3), flush=True)
S = {k: st.mean(r[k] for r in res) for k in ['rel_norm_dW_over_W0', 'rel_err', 'cos', 'frac_elems_unchanged', 'frac_dW_below_half_ulp']}; S['n_matrices'] = len(res)
json.dump({'summary': S, 'per_matrix': res}, open(out, 'w'), indent=1); print(S)
