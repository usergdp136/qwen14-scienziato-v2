# Figure per l'articolo (IT/EN): pipeline, barre con IC Wilson, scelta scala su dev, loss di training.
import json, math, sys, pathlib
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
R = pathlib.Path(sys.argv[1]); OUT = pathlib.Path(sys.argv[2]); OUT.mkdir(exist_ok=True, parents=True)
def wilson(k, n, z=1.96):
    p = k / n; den = 1 + z * z / n; c = (p + z * z / (2 * n)) / den; h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den; return c - h, c + h
def accs(v, su):
    k = n = 0
    for s in su:
        d = json.load(open(R / 'results' / f'{v}__{s}.json'))
        for r in d['rows']: k += int(bool(r['ok'] if 'ok' in r else r['prompt_level_strict_acc'])); n += 1
    return k, n
T = {'it': dict(vars=['base', 'ft (s=1.0)', 's=0.85', 'merged (bf16)'], sets=['ARC-it\n(n=1132)', 'GSM8K held-out\n(n=919)', 'IFEval held-out\n(n=341)', 'Identità / Identity\n(n=50, keyword)'], y='Accuratezza (%)  ± IC 95% Wilson',
          t='Test: base vs ft vs scala 0.85 vs merge fisico bf16', dev_t='Scelta della scala sul DEV (regola R1 pre-registrata)', x='scala di merge s', loss_t='Loss di training (media mobile su 20 passi)', step='passo', loss='loss',
          ident='identità dev (n=40)', gsm='GSM8K dev (n=400)', ife='IFEval dev (n=200)', thr='soglia identità 95%', chosen='scala scelta'),
     'en': dict(vars=['base', 'ft (s=1.0)', 's=0.85', 'merged (bf16)'], sets=['ARC-it\n(n=1132)', 'GSM8K held-out\n(n=919)', 'IFEval held-out\n(n=341)', 'Identity\n(n=50, keyword)'], y='Accuracy (%)  ± 95% Wilson CI',
          t='Test: base vs ft vs scale 0.85 vs physical bf16 merge', dev_t='Scale selection on DEV (pre-registered rule R1)', x='merge scale s', loss_t='Training loss (20-step moving average)', step='step', loss='loss',
          ident='identity dev (n=40)', gsm='GSM8K dev (n=400)', ife='IFEval dev (n=200)', thr='identity threshold 95%', chosen='chosen scale')}
SUITES = [['arc_test'], ['gsm_test', 'gsm_rest'], ['ife_test', 'ife_rest'], ['ident_test']]; VAR = ['base', 'ft', 's085', 'merged']; COL = ['#9aa5b1', '#2a7de1', '#f4a261', '#e76f51']
for lg, L in T.items():
    fig, ax = plt.subplots(figsize=(9, 4.6)); w = 0.2
    for i, v in enumerate(VAR):
        xs, ys, lo, hi = [], [], [], []
        for j, su in enumerate(SUITES):
            k, n = accs(v, su); a = 100 * k / n; l, h = wilson(k, n); xs.append(j + (i - 1.5) * w); ys.append(a); lo.append(a - 100 * l); hi.append(100 * h - a)
        ax.bar(xs, ys, w, yerr=[lo, hi], capsize=2.5, color=COL[i], label=L['vars'][i], edgecolor='white')
        for x, y, h in zip(xs, ys, hi): ax.text(x, (2 if y > 30 else y + h + 1.5), f'{y:.1f}', ha='center', va='bottom', fontsize=6.5, rotation=90, color='black')
    ax.set_xticks(range(4)); ax.set_xticklabels(L['sets'], fontsize=8); ax.set_ylim(0, 108); ax.set_ylabel(L['y']); ax.set_title(L['t'], fontsize=10); ax.legend(ncol=4, fontsize=8, loc='upper center', bbox_to_anchor=(0.5, -0.17), frameon=False); ax.grid(axis='y', alpha=.3)
    fig.tight_layout(); fig.savefig(OUT / f'fig_bars_{lg}.png', dpi=200); plt.close(fig)
    # scelta scala
    sel = json.load(open(R / 'select.json')); sc = [0.0] + [t['scale'] for t in sel['table']]
    ident = [100 * sel['base']['ident_dev']] + [100 * t['ident_dev'] for t in sel['table']]; gsm = [100 * sel['base']['gsm_dev']] + [100 * t['gsm_dev'] for t in sel['table']]; ife = [100 * sel['base']['ife_dev']] + [100 * t['ife_dev'] for t in sel['table']]
    fig, ax = plt.subplots(1, 2, figsize=(9, 3.8))
    ax[0].plot(sc, ident, 'o-', color='#2a7de1', label=L['ident']); ax[0].axhline(95, color='#2a7de1', ls='--', lw=1, label=L['thr']); ax[0].axvline(1.0, color='green', ls=':', label=L['chosen'] + ' = 1.0')
    ax[0].set_xlabel(L['x']); ax[0].set_ylabel('%'); ax[0].set_ylim(-3, 105); ax[0].legend(fontsize=7, loc='center left'); ax[0].grid(alpha=.3)
    ax[1].plot(sc, gsm, 's-', color='#e76f51', label=L['gsm']); ax[1].plot(sc, ife, '^-', color='#2a9d8f', label=L['ife'])
    ax[1].axhline(gsm[0] - 1, color='#e76f51', ls='--', lw=1); ax[1].axhline(ife[0] - 1, color='#2a9d8f', ls='--', lw=1); ax[1].set_xlabel(L['x']); ax[1].set_ylim(83, 98); ax[1].legend(fontsize=7, loc='lower right'); ax[1].grid(alpha=.3)
    ax[1].set_title('— : base − 1 pt', fontsize=8); fig.suptitle(L['dev_t'], fontsize=10); fig.tight_layout(); fig.savefig(OUT / f'fig_scale_{lg}.png', dpi=200); plt.close(fig)
    # loss
    m = [json.loads(l) for l in open(R / 'metrics.jsonl')]; ls = [x['loss'] for x in m]; k = 20; ma = [sum(ls[max(0, i - k + 1):i + 1]) / len(ls[max(0, i - k + 1):i + 1]) for i in range(len(ls))]
    fig, ax = plt.subplots(figsize=(7, 3.4)); ax.plot([x['step'] for x in m], ls, color='#c7d3e0', lw=.8); ax.plot([x['step'] for x in m], ma, color='#2a7de1', lw=1.8)
    ax2 = ax.twinx(); ax2.plot([x['step'] for x in m], [x['learning_rate'] for x in m], color='#f4a261', lw=1, ls='--'); ax2.set_ylabel('LR', color='#f4a261')
    ax.set_xlabel(L['step']); ax.set_ylabel(L['loss']); ax.set_title(L['loss_t'], fontsize=10); ax.grid(alpha=.3); fig.tight_layout(); fig.savefig(OUT / f'fig_loss_{lg}.png', dpi=200); plt.close(fig)
# pipeline
P = {'it': ['Dati: prompt\naya_ita, oasst1,\nGSM8K-train, prompt\ngenerati da Qwen3-14B', 'Auto-distillazione\nQwen3-14B bf16 (vLLM)\nrisposte proprie, T=0.7\n+ scheda identità nascosta', 'Filtri + dedup\nno codice, verifica GSM8K,\ncheck vincoli IFEval\n8601 train + 299 val', 'LoRA r=16, α=16\nLR 3e-5, 1 epoca\n538 passi, bf16\nΔW=(α/r)BA', 'Scale di merge\ns∈{0.3,0.5,0.7,0.85,1.0}\nW=W0+s·ΔW\nregola R1 su DEV', 'Valutazione TEST\nvLLM bf16, greedy\nARC-it, GSM8K, IFEval,\nidentità + McNemar'],
     'en': ['Data: prompts\naya_ita, oasst1,\nGSM8K-train, prompts\ngenerated by Qwen3-14B', 'Self-distillation\nQwen3-14B bf16 (vLLM)\nits own answers, T=0.7\n+ hidden identity card', 'Filters + dedup\nno code, GSM8K verified,\nIFEval-style checks\n8601 train + 299 val', 'LoRA r=16, α=16\nLR 3e-5, 1 epoch\n538 steps, bf16\nΔW=(α/r)BA', 'Merge scales\ns∈{0.3,0.5,0.7,0.85,1.0}\nW=W0+s·ΔW\nrule R1 on DEV', 'TEST evaluation\nvLLM bf16, greedy\nARC-it, GSM8K, IFEval,\nidentity + McNemar']}
for lg, boxes in P.items():
    fig, ax = plt.subplots(figsize=(10, 5.2)); ax.set_xlim(0, 3 * 3.4); ax.set_ylim(0, 2.3); ax.axis('off')
    fc = ['#e8f1fb', '#e8f1fb', '#e8f1fb', '#fdf0e1', '#fdf0e1', '#e6f4ee']
    pos = [(0, 1.25), (1, 1.25), (2, 1.25), (0, 0.1), (1, 0.1), (2, 0.1)]
    for i, t in enumerate(boxes):
        x = pos[i][0] * 3.4 + 0.1; y = pos[i][1]
        ax.add_patch(FancyBboxPatch((x, y), 2.7, 0.95, boxstyle='round,pad=0.02,rounding_size=0.08', fc=fc[i], ec='#2a4d7a', lw=1.4))
        ax.text(x + 1.35, y + 0.475, t, ha='center', va='center', fontsize=11)
    for i in (0, 1): ax.add_patch(FancyArrowPatch((i * 3.4 + 2.83, 1.725), (i * 3.4 + 3.47, 1.725), arrowstyle='-|>', mutation_scale=16, color='#2a4d7a'))
    ax.add_patch(FancyArrowPatch((6.9 + 1.45, 1.22), (0.1 + 1.35, 1.08), connectionstyle='angle3,angleA=-90,angleB=0', arrowstyle='-|>', mutation_scale=16, color='#2a4d7a', lw=1))
    for i in (3, 4): ax.add_patch(FancyArrowPatch(((i - 3) * 3.4 + 2.83, 0.575), ((i - 3) * 3.4 + 3.47, 0.575), arrowstyle='-|>', mutation_scale=16, color='#2a4d7a'))
    fig.tight_layout(); fig.savefig(OUT / f'fig_pipeline_{lg}.png', dpi=170); plt.close(fig)
print('ok')
