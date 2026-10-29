# Valuta un GGUF servito da llama-server (o Ollama, endpoint OpenAI-compatibile) su identità (stesso scorer di pod/score.py) e GSM8K (test 150).
# Uso: python eval_gguf.py <url_base> <cartella_sets> <out.json> [suite,...]   es.  http://127.0.0.1:8099 R2/sets out.json ident_test,gsm_test
import sys, json, re, time, requests
from concurrent.futures import ThreadPoolExecutor
URL, SETS, OUT = sys.argv[1], sys.argv[2], sys.argv[3]; SU = (sys.argv[4] if len(sys.argv) > 4 else 'ident_test').split(',')
MAXTOK = {'gsm': 512, 'ident': 256}
def chat(msgs, n):
    b = {'messages': msgs, 'max_tokens': n, 'temperature': 0, 'top_p': 1.0, 'seed': 1234, 'chat_template_kwargs': {'enable_thinking': False}}
    for a in range(4):
        try:
            r = requests.post(URL + '/v1/chat/completions', json=b, timeout=3600); r.raise_for_status(); j = r.json(); return j['choices'][0]['message']['content'] or '', j['choices'][0]['finish_reason']
        except Exception as e: print('retry', a, e, flush=True); time.sleep(5)
    raise RuntimeError('fail')
R = {}
for su in SU:
    kind = su.split('_')[0]; items = [json.loads(l) for l in open(f'{SETS}/{su}.jsonl')]; t0 = time.time()
    def f(it):
        msgs = it['msgs'] if kind == 'ident' else [{'role': 'user', 'content': it['prompt']}]
        o, fin = chat(msgs, MAXTOK[kind])
        if kind == 'ident':
            l = o.lower(); own = any(k in l for k in ('arianna labs', 'gdp', 'del papa', 'scienziato dilettante'))
            wrong = bool(re.search(r'\b(chatgpt|openai|claude|anthropic|gemini|llama|meta ai|mistral)\b', l)) and not re.search(r"non sono|not |n't|né", l)
            return {'id': it['id'], 'q': it['msgs'][-1]['content'], 'answer': o, 'ok': own and not wrong}
        nums = re.findall(r'-?\d[\d,]*\.?\d*', o.split('answer is')[-1] if 'answer is' in o else o); p = None
        if nums:
            try: p = float(nums[-1].replace(',', '').rstrip('.'))
            except Exception: p = None
        return {'id': it['id'], 'pred': p, 'gold': it['gold'], 'ok': bool(p is not None and abs(p - it['gold']) < 1e-6), 'out': o[-800:], 'finish': fin}
    with ThreadPoolExecutor(4) as ex: rows = list(ex.map(f, items))
    R[su] = {'n': len(rows), 'acc': sum(r['ok'] for r in rows) / len(rows), 'elapsed_s': round(time.time() - t0), 'rows': rows}; print(su, R[su]['acc'], R[su]['elapsed_s'], 's', flush=True)
    json.dump(R, open(OUT, 'w'), ensure_ascii=False, indent=1)
