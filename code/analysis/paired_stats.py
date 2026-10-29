# Statistiche appaiate sui file per-item results/<variante>__<suite>.json (stesso item = stessa chiave).
import json, math, pathlib, sys
R = pathlib.Path(sys.argv[1] if len(sys.argv) > 1 else 'results')
def rows(v, suites):
    d = {}
    for s in suites:
        for r in json.load(open(R / f'{v}__{s}.json'))['rows']:
            d[(s, r['id'])] = int(bool(r['ok'] if 'ok' in r else r['prompt_level_strict_acc']))
    return d
def mcnemar_exact(b, c):               # b = rotte (1->0), c = corrette in più (0->1)
    n = b + c
    if n == 0: return 1.0
    k = min(b, c); p = sum(math.comb(n, i) for i in range(k + 1)) / 2 ** n
    return min(1.0, 2 * p)
def paired_ci(b, c, N, z=1.96):        # IC di Wald per la differenza di proporzioni appaiate
    d = (c - b) / N; var = (b + c - (c - b) ** 2 / N) / N ** 2; se = math.sqrt(max(var, 0)); return d, d - z * se, d + z * se
def wilson(k, n, z=1.96):
    p = k / n; den = 1 + z * z / n; ctr = (p + z * z / (2 * n)) / den; h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den; return ctr - h, ctr + h
SETS = {'ARC-it test': ['arc_test'], 'GSM8K held-out (test+rest)': ['gsm_test', 'gsm_rest'], 'IFEval held-out (test+rest)': ['ife_test', 'ife_rest'],
        'Identita test': ['ident_test'], 'GSM8K dev': ['gsm_dev'], 'IFEval dev': ['ife_dev'], 'Identita dev': ['ident_dev']}
out = {}
for name, su in SETS.items():
    base = rows('base', su); N = len(base); out[name] = {'n': N, 'base_acc': sum(base.values()) / N, 'base_wilson': wilson(sum(base.values()), N)}
    for v in ['ft', 's085', 'merged']:
        try: d = rows(v, su)
        except FileNotFoundError: continue
        b = sum(1 for k in base if base[k] == 1 and d[k] == 0); c = sum(1 for k in base if base[k] == 0 and d[k] == 1)
        diff, lo, hi = paired_ci(b, c, N); k = sum(d.values())
        out[name][v] = {'acc': k / N, 'wilson': wilson(k, N), 'fixed': c, 'broken': b, 'delta_pt': 100 * diff, 'delta_ci95_pt': (100 * lo, 100 * hi), 'mcnemar_p': mcnemar_exact(b, c)}
json.dump(out, open('paired_stats.json', 'w'), indent=1)
for n, e in out.items():
    print(f"{n} (n={e['n']}) base {100*e['base_acc']:.1f}% Wilson [{100*e['base_wilson'][0]:.1f},{100*e['base_wilson'][1]:.1f}]")
    for v in ['ft', 's085', 'merged']:
        if v in e:
            x = e[v]; print(f"   {v:7s} {100*x['acc']:.1f}%  fixed {x['fixed']} broken {x['broken']}  d={x['delta_pt']:+.2f} pt CI[{x['delta_ci95_pt'][0]:+.2f},{x['delta_ci95_pt'][1]:+.2f}]  p={x['mcnemar_p']:.4f}")
