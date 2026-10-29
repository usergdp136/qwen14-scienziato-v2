# P2: LoRA gentile su Qwen3-14B in bf16 (NIENTE 4-bit). r16 a16 drop .05, tutti i lineari, LR 3e-5 cosine warmup 3%, 1 epoch, batch effettivo 16, seq 2048.
import os, sys, json, math, time, random
os.environ.setdefault('TOKENIZERS_PARALLELISM', 'false')
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import LoraConfig, get_peft_model
from common import *
LR = float(os.environ.get('LR', 3e-5)); IDX = int(os.environ.get('IDENT_X', 1)); OUT = os.environ.get('OUT', W + '/adapter'); MAXLEN = 2048; BS = 16; MB_TOK = 8192; ETA_MAX_H = 2.2
tok = AutoTokenizer.from_pretrained(W + '/base'); rows = rjsonl(W + '/data/train.jsonl'); val = rjsonl(W + '/data/val.jsonl')
if IDX > 1: rows = rows + [r for r in rows if r['block'] == 'D'] * (IDX - 1)
def enc(r):
    msgs = r['msgs']; p = qwen_prompt(tok, msgs[:-1]); a = msgs[-1]['content'] + '<|im_end|>'
    pi = tok(p, add_special_tokens=False)['input_ids']; ai = tok(a, add_special_tokens=False)['input_ids']
    return pi + ai, [-100] * len(pi) + ai
# verifica template: training (prompt+risposta) == apply_chat_template completo
bad = 0
for r in rows[:300] + [x for x in rows if x['block'] == 'D'][:60]:
    full = tok.apply_chat_template(r['msgs'], tokenize=False, enable_thinking=False); mine = qwen_prompt(tok, r['msgs'][:-1]) + r['msgs'][-1]['content'] + '<|im_end|>\n'
    if full != mine: bad += 1; 
    if full != mine and bad <= 2: print('TEMPLATE MISMATCH\n', repr(full[-300:]), '\n', repr(mine[-300:]), flush=True)
print('template check: mismatches', bad, flush=True)
json.dump({'checked': 360, 'mismatches': bad}, open(W + '/template_check.json', 'w'))
if bad > 0: phase('P2', 0, 'ERRORE: template training != eval (%d)' % bad); sys.exit(5)
data = []
for r in rows:
    ids, lab = enc(r)
    if len(ids) <= MAXLEN: data.append((ids, lab, r['block']))
vdata = [enc(r) for r in val]; vdata = [d for d in vdata if len(d[0]) <= MAXLEN]
random.Random(1234).shuffle(data)
nsteps = math.ceil(len(data) / BS); tot_tok = sum(len(d[0]) for d in data)
print('train seq', len(data), 'steps', nsteps, 'tokens', tot_tok, 'target tokens', sum(sum(1 for x in d[1] if x != -100) for d in data), flush=True)
model = AutoModelForCausalLM.from_pretrained(W + '/base', dtype=torch.bfloat16, attn_implementation='sdpa', device_map={'': 0})
model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant': False}); model.enable_input_require_grads(); model.config.use_cache = False
cfg = LoraConfig(r=16, lora_alpha=16, lora_dropout=0.05, target_modules=['q_proj', 'k_proj', 'v_proj', 'o_proj', 'gate_proj', 'up_proj', 'down_proj'], task_type='CAUSAL_LM', bias='none')
model = get_peft_model(model, cfg); model.print_trainable_parameters()
params = [p for p in model.parameters() if p.requires_grad]
opt = torch.optim.AdamW(params, lr=LR, weight_decay=0.0, betas=(0.9, 0.999), fused=True); warm = max(1, int(0.03 * nsteps))
def lr_at(s): return LR * (s + 1) / warm if s < warm else LR * 0.5 * (1 + math.cos(math.pi * (s - warm) / max(1, nsteps - warm)))
def collate(chunk):
    L = max(len(c[0]) for c in chunk); pad = tok.pad_token_id or 0
    ids = torch.tensor([c[0] + [pad] * (L - len(c[0])) for c in chunk]).cuda(); lab = torch.tensor([c[1] + [-100] * (L - len(c[1])) for c in chunk]).cuda()
    att = torch.tensor([[1] * len(c[0]) + [0] * (L - len(c[0])) for c in chunk]).cuda(); return ids, lab, att
def micro(batch):
    b = sorted(batch, key=lambda d: len(d[0])); out = []; cur = []
    for d in b:
        if cur and (len(cur) + 1) * len(d[0]) > MB_TOK: out.append(cur); cur = []
        cur.append(d)
    if cur: out.append(cur)
    return out
def loss_sum(chunk):
    ids, lab, att = collate(chunk)
    with torch.autocast('cuda', dtype=torch.bfloat16): lg = model(input_ids=ids, attention_mask=att).logits
    return torch.nn.functional.cross_entropy(lg[:, :-1].float().reshape(-1, lg.size(-1)), lab[:, 1:].reshape(-1), ignore_index=-100, reduction='sum')
mf = open(W + '/metrics.jsonl', 'a'); t0 = time.time(); seen_tok = 0; model.train()
phase('P2', 0, 'training avviato: %d seq, %d step' % (len(data), nsteps), 'P3 selezione scala')
for step in range(nsteps):
    batch = data[step * BS:(step + 1) * BS]; ntgt = sum(sum(1 for x in d[1][1:] if x != -100) for d in batch); tl = 0.0
    for g in opt.param_groups: g['lr'] = lr_at(step)
    for ch in micro(batch):
        l = loss_sum(ch) / ntgt; l.backward(); tl += l.item()
    gn = torch.nn.utils.clip_grad_norm_(params, 1.0).item(); opt.step(); opt.zero_grad(set_to_none=True)
    seen_tok += sum(len(d[0]) for d in batch); el = time.time() - t0; tps = seen_tok / el
    mf.write(json.dumps({'step': step + 1, 'max_steps': nsteps, 'ts': time.time(), 'elapsed_s': round(el, 1), 'tok_s_upper': round(tps, 1), 'loss': tl, 'grad_norm': gn, 'learning_rate': lr_at(step), 'epoch': (step + 1) / nsteps}) + '\n'); mf.flush()
    eta = (tot_tok - seen_tok) / tps
    if (step + 1) % 5 == 0 or step < 3: phase('P2', round(100 * (step + 1) / nsteps, 1), 'step %d/%d loss %.3f %.0f tok/s' % (step + 1, nsteps, tl, tps), 'P3 selezione scala', eta_s=eta)
    if step + 1 == 30:
        eta_total_h = tot_tok / tps / 3600; print('SMOKE30: tok/s %.0f -> training totale %.2f h' % (tps, eta_total_h), flush=True)
        json.dump({'tok_s': tps, 'eta_total_h': eta_total_h, 'loss30': tl}, open(W + '/smoke.json', 'w'))
        if eta_total_h > ETA_MAX_H: phase('P2', 5, 'ABORT: ETA training %.2f h > %.1f h' % (eta_total_h, ETA_MAX_H)); open(W + '/ABORT', 'w').write('eta'); sys.exit(3)
    if (step + 1) % 200 == 0: model.save_pretrained(OUT + '_ckpt%d' % (step + 1))
model.save_pretrained(OUT); print('adapter salvato', OUT, 'elapsed', round(time.time() - t0), flush=True)
model.eval(); tl = 0; tn = 0
with torch.no_grad():
    for i in range(0, len(vdata), 4):
        ch = vdata[i:i + 4]; tl += loss_sum(ch).item(); tn += sum(sum(1 for x in d[1][1:] if x != -100) for d in ch)
json.dump({'val_loss': tl / tn, 'n_val': len(vdata), 'train_elapsed_s': time.time() - t0, 'steps': nsteps, 'tok_s': seen_tok / (time.time() - t0)}, open(W + '/train_summary.json', 'w'))
print('val loss', tl / tn, flush=True); phase('P2', 100, 'training finito, val loss %.3f' % (tl / tn), 'P3 selezione scala'); open(W + '/P2_DONE', 'w').write('ok')
