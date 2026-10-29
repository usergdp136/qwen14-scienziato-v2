# Regola R1 (pre-registrata, SOLO dev): identity dev(40) >= 95% (>=38/40), GSM8K dev(400) >= base-1pt, IFEval dev(200, prompt-level strict) >= base-1pt. Si sceglie la scala PIU' PICCOLA che qualifica.
import json, pathlib
R = pathlib.Path('/workspace/results'); L = [('s03', 0.3), ('s05', 0.5), ('s07', 0.7), ('s085', 0.85), ('ft', 1.0)]
def acc(v, s): return json.load(open(R / f'{v}__{s}.json'))['acc']
b = {s: acc('base', s) for s in ('gsm_dev', 'ife_dev', 'ident_dev')}; table = []; chosen = None
for n, sc in L:
    a = {s: acc(n, s) for s in ('gsm_dev', 'ife_dev', 'ident_dev')}
    q = {'identity': a['ident_dev'] >= 0.95 - 1e-9, 'gsm8k': a['gsm_dev'] >= b['gsm_dev'] - 0.01 - 1e-9, 'ifeval': a['ife_dev'] >= b['ife_dev'] - 0.01 - 1e-9}
    ok = all(q.values()); table.append({'name': n, 'scale': sc, **a, 'checks': q, 'qualifies': ok})
    if ok and chosen is None: chosen = {'name': n, 'scale': sc}
json.dump({'rule': 'R1 dev only', 'base': b, 'table': table, 'chosen': chosen}, open('/workspace/select.json', 'w'), indent=1)
print('BASE', b)
for t in table: print(t['name'], t['scale'], round(t['ident_dev'], 3), round(t['gsm_dev'], 4), round(t['ife_dev'], 4), t['qualifies'], t['checks'])
print('CHOSEN', chosen)
