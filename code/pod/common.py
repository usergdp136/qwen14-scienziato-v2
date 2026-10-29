import json, os, re, time, random, hashlib, requests
from concurrent.futures import ThreadPoolExecutor
W = '/workspace'
def phase(code, pct, detail='', nxt='', risultato=None, eta_s=None):
    d = {'phase': code, 'pct': pct, 'detail': detail, 'next': nxt, 'ts': time.time()}
    if eta_s: d['eta_ts'] = time.time() + eta_s
    if risultato: d['risultato'] = risultato
    t = W + '/phase.json.tmp'; json.dump(d, open(t, 'w'), ensure_ascii=False); os.replace(t, W + '/phase.json')
def norm(s): return re.sub(r'\W+', ' ', s.lower()).strip()
def toks(s): return set(norm(s).split())
def jac(a, b):
    if not a or not b: return 0.0
    return len(a & b) / len(a | b)
def chat(msgs, max_tokens=1500, temperature=0.7, top_p=0.8, top_k=20, model='base', port=8000):
    body = {'model': model, 'messages': msgs, 'max_tokens': max_tokens, 'temperature': temperature, 'top_p': top_p, 'top_k': top_k, 'chat_template_kwargs': {'enable_thinking': False}}
    err = ''
    for a in range(5):
        try:
            r = requests.post('http://127.0.0.1:%d/v1/chat/completions' % port, json=body, timeout=1800)
            if r.status_code == 200:
                j = r.json(); c = j['choices'][0]; return c['message']['content'] or '', c['finish_reason'], j['usage']['completion_tokens']
            err = '%s %s' % (r.status_code, r.text[:200])
        except Exception as e: err = str(e)
        time.sleep(3 * (a + 1))
    return '', 'error', 0
def pmap(fn, items, conc=192, label='', every=0, cb=None):
    out = [None] * len(items); done = [0]; t0 = time.time()
    def w(i):
        out[i] = fn(items[i]); done[0] += 1
        if cb and done[0] % max(1, every or 200) == 0: cb(done[0], len(items))
    with ThreadPoolExecutor(conc) as ex: list(ex.map(w, range(len(items))))
    print(label, len(items), 'in', round(time.time() - t0), 's', flush=True); return out
def wjsonl(p, rows):
    with open(p, 'w') as f:
        for r in rows: f.write(json.dumps(r, ensure_ascii=False) + '\n')
def rjsonl(p): return [json.loads(l) for l in open(p)]
def qwen_prompt(tok, msgs):
    """STESSA funzione per training ed eval: prompt = template Qwen3 con enable_thinking=False + generation prompt (include <think>\\n\\n</think>\\n\\n vuoto)."""
    return tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True, enable_thinking=False)
