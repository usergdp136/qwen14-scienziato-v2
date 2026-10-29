# Generazione deterministica (temp 0, seed 1234, no thinking, template del tokenizer di training) via vLLM /v1/completions.
# Uso: gen.py <varianti,> <suite,>   -> /workspace/results/<variante>__<suite>.gen.json (risposte grezze, un item per riga; si riprende se esiste)
import sys, json, time, os, pathlib, requests
from concurrent.futures import ThreadPoolExecutor
from transformers import AutoTokenizer
VARS, SUITES = sys.argv[1].split(','), sys.argv[2].split(','); CONC = int(os.environ.get('CONC', '96'))
tok = AutoTokenizer.from_pretrained('/workspace/base'); RES = pathlib.Path('/workspace/results'); SETS = pathlib.Path('/workspace/sets')
URL = 'http://127.0.0.1:%s/v1/completions' % os.environ.get('PORT', '8000')
MAXTOK = {'gsm': 512, 'ife': 1024, 'ident': 256, 'arc': 1}
def ids(msgs): return tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True, enable_thinking=False)   # identico a qwen_prompt() del training
def call(model, msgs, n, **kw):
    text = ids(msgs); pid = tok(text, add_special_tokens=False)['input_ids']
    body = {'model': model, 'prompt': pid, 'max_tokens': n, 'temperature': 0, 'top_p': 1.0, 'seed': 1234, **kw}
    for a in range(5):
        try:
            r = requests.post(URL, json=body, timeout=1800)
            if r.status_code == 200: return r.json(), len(pid)
            err = f'{r.status_code} {r.text[:300]}'
        except Exception as e: err = str(e)
        print('retry', a, err, flush=True); time.sleep(5 * (a + 1))
    raise RuntimeError(err)
def run(var, suite):
    out = RES / f'{var}__{suite}.gen.json'
    if out.exists(): print('skip', out.name); return
    items = [json.loads(l) for l in open(SETS / f'{suite}.jsonl')]; kind = suite.split('_')[0]; t0 = time.time()
    def f(it):
        if kind == 'arc':
            j, pl = call(var, [{'role': 'user', 'content': it['prompt']}], 1, logprobs=20); c = j['choices'][0]
            tops = ((c.get('logprobs') or {}).get('top_logprobs') or [{}])[0]
            return {'id': it['id'], 'gen': c['text'], 'top': tops, 'prompt_tokens': pl}
        msgs = it['msgs'] if kind == 'ident' else [{'role': 'user', 'content': it['prompt']}]
        j, pl = call(var, msgs, MAXTOK[kind]); c = j['choices'][0]
        return {'id': it['id'], 'out': c['text'], 'finish': c['finish_reason'], 'out_tok': j['usage']['completion_tokens'], 'prompt_tokens': pl}
    with ThreadPoolExecutor(CONC) as ex: rows = list(ex.map(f, items))
    json.dump({'variant': var, 'suite': suite, 'n': len(rows), 'elapsed_s': round(time.time() - t0, 1), 'rows': rows}, open(str(out) + '.tmp', 'w'), ensure_ascii=False)
    os.replace(str(out) + '.tmp', out); print(var, suite, len(rows), round(time.time() - t0), 's', flush=True)
for s in SUITES:
    for v in VARS: run(v, s)
