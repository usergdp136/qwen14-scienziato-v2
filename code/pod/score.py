# Punteggio (stessa logica di qwen36_27b_eval/bench_api.py). Venv2 (lm_eval per IFEval). Uso: score.py  -> results/<v>__<suite>.json + summary.json
import json, re, pathlib, glob, os
RES = pathlib.Path('/workspace/results'); SETS = pathlib.Path('/workspace/sets'); L = 'ABCDE'
def load(s): return {r['id']: r for r in map(json.loads, open(SETS / f'{s}.jsonl'))}
def score(var, suite):
    g = json.load(open(RES / f'{var}__{suite}.gen.json')); items = load(suite); kind = suite.split('_')[0]; rows = []
    if kind == 'arc':
        for r in g['rows']:
            it = items[r['id']]; sc = {}
            for t, v in (r['top'] or {}).items():
                t = t.strip()
                if t in L[:it['k']]: sc[t] = max(sc.get(t, -1e9), v)
            gt = (r['gen'] or '').strip()[:1]; pred = max(sc, key=sc.get) if sc else (gt if gt in L[:it['k']] else None)
            rows.append({'id': r['id'], 'pred': pred, 'gold': it['gold'], 'ok': pred == it['gold'], 'via': 'logprob' if sc else 'gen', 'gen': gt})
    elif kind == 'gsm':
        for r in g['rows']:
            it = items[r['id']]; o = r['out']; gold = it['gold']
            nums = re.findall(r'-?\d[\d,]*\.?\d*', o.split('answer is')[-1] if 'answer is' in o else o); p = None
            if nums:
                try: p = float(nums[-1].replace(',', '').rstrip('.'))
                except Exception: p = None
            rows.append({'id': r['id'], 'pred': p, 'gold': gold, 'ok': bool(p is not None and abs(p - gold) < 1e-6), 'out': o[-1500:], 'out_tok': r['out_tok'], 'finish': r['finish']})
    elif kind == 'ident':
        for r in g['rows']:
            it = items[r['id']]; o = r['out']; l = o.lower()
            own = any(k in l for k in ('arianna labs', 'gdp', 'del papa', 'scienziato dilettante')); wrong = bool(re.search(r'\b(chatgpt|openai|claude|anthropic|gemini|llama|meta ai|mistral)\b', l)) and not re.search(r"non sono|not |n't|né", l)
            rows.append({'id': r['id'], 'q': it['msgs'][-1]['content'], 'answer': o, 'ok': own and not wrong, 'out_tok': r['out_tok']})
    elif kind == 'ife':
        from lm_eval.tasks.ifeval import utils as U
        for r in g['rows']:
            it = items[r['id']]; res = U.process_results({'key': it['key'], 'prompt': it['prompt'], 'instruction_id_list': it['instruction_id_list'], 'kwargs': it['kwargs']}, [r['out']])
            rows.append({'id': r['id'], 'key': it['key'], 'resp': r['out'][:3000], **res, 'out_tok': r['out_tok'], 'finish': r['finish']})
    S = {'variant': var, 'suite': suite, 'n': len(rows)}
    if kind == 'ife':
        for k in ['prompt_level_strict_acc', 'prompt_level_loose_acc']: S[k] = round(sum(x[k] for x in rows) / len(rows), 4)
        for k in ['inst_level_strict_acc', 'inst_level_loose_acc']: v = [y for x in rows for y in x[k]]; S[k] = round(sum(v) / len(v), 4)
        S['acc'] = S['prompt_level_strict_acc']
    else: S['acc'] = round(sum(r['ok'] for r in rows) / len(rows), 4)
    if kind == 'arc': S['via_logprob'] = sum(r['via'] == 'logprob' for r in rows)
    if kind == 'gsm': S['truncated'] = sum(r['finish'] == 'length' for r in rows)
    S['rows'] = rows; json.dump(S, open(RES / f'{var}__{suite}.json', 'w'), indent=1, ensure_ascii=False); return {k: v for k, v in S.items() if k != 'rows'}
summ = {}
for f in sorted(glob.glob(str(RES / '*.gen.json'))):
    var, suite = pathlib.Path(f).name[:-9].split('__'); summ[f'{var}__{suite}'] = score(var, suite)
json.dump(summ, open(RES / 'summary.json', 'w'), indent=1)
for k, v in summ.items(): print(k, v['n'], v['acc'])
