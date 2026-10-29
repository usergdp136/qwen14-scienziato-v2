# Statistiche finali: accuratezze, CI Wilson 95%, McNemar esatto vs base (appaiato sugli stessi item). Uso: final_stats.py var1,var2,...
import json, sys, math, pathlib
R = pathlib.Path('/workspace/results'); V = sys.argv[1].split(',')
SUITES = {'arc': ['arc_test'], 'gsm': ['gsm_test', 'gsm_dev', 'gsm_rest'], 'ife': ['ife_test', 'ife_dev', 'ife_rest'], 'ident': ['ident_test', 'ident_dev']}
def rows(v, suites):
    d = {}
    for s in suites:
        p = R / f'{v}__{s}.json'
        if not p.exists(): return None
        for r in json.load(open(p))['rows']: d[(s, r['id'])] = int(bool(r['ok'] if 'ok' in r else r['prompt_level_strict_acc']))
    return d
def wilson(k, n):
    if n == 0: return (0, 0)
    p = k / n; z = 1.96; den = 1 + z * z / n; c = (p + z * z / (2 * n)) / den; h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den; return (round(c - h, 4), round(c + h, 4))
def mcn(a, b):
    n01 = sum(1 for k in a if a[k] == 0 and b[k] == 1); n10 = sum(1 for k in a if a[k] == 1 and b[k] == 0); n = n01 + n10
    if n == 0: return 1.0, n01, n10
    k = min(n01, n10); p = sum(math.comb(n, i) for i in range(0, k + 1)) / 2 ** n; return min(1.0, 2 * p), n01, n10
sets = {'ARC-it (1132)': ('arc', None), 'GSM8K full (1319)': ('gsm', None), 'GSM8K held-out (919: test+rest)': ('gsm', ['gsm_test', 'gsm_rest']), 'GSM8K dev (400)': ('gsm', ['gsm_dev']),
        'IFEval full (541)': ('ife', None), 'IFEval held-out (341: test+rest)': ('ife', ['ife_test', 'ife_rest']), 'IFEval dev (200)': ('ife', ['ife_dev']), 'Identity test (50)': ('ident', ['ident_test']), 'Identity dev (40)': ('ident', ['ident_dev'])}
out = {}; md = ['| Set | ' + ' | '.join(V) + ' |', '|---|' + '---|' * len(V)]
for name, (k, sub) in sets.items():
    suites = sub or SUITES[k]; cells = []; base = rows('base', suites); out[name] = {}
    for v in V:
        d = rows(v, suites)
        if d is None: cells.append('—'); continue
        n = len(d); s = sum(d.values()); ci = wilson(s, n); e = {'acc': round(s / n, 4), 'n': n, 'ci95': ci}
        if v != 'base' and base:
            p, n01, n10 = mcn(base, d); e.update(delta_pt=round(100 * (s / n - sum(base.values()) / n), 2), mcnemar_p=round(p, 4), fixed=n01, broken=n10)
        out[name][v] = e; cells.append('%.1f%%' % (100 * e['acc']) + ('' if v == 'base' or 'delta_pt' not in e else ' (%+.1f, p=%.3f)' % (e['delta_pt'], e['mcnemar_p'])))
    md.append('| ' + name + ' | ' + ' | '.join(cells) + ' |')
json.dump(out, open('/workspace/final_stats.json', 'w'), indent=1); open('/workspace/final_table.md', 'w').write('\n'.join(md) + '\n'); print('\n'.join(md))
