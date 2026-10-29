# P1: auto-distillazione. Qwen3-14B (vLLM bf16, no thinking, T0.7 top-p .8 top-k 20) scrive le PROPRIE risposte. Server vLLM su :8000 (model 'base').
import glob, sys, collections
import pandas as pd
from common import *
import constraints as C, identity as I
rng = random.Random(3407); D = W + '/data'; SETS = W + '/sets'
BASE_REV = '40c069824f4251a91eefaf281ebe4c544efd3e18'
CODE = re.compile(r"```|\b(python|java|javascript|c\+\+|c#|sql|html|css|php|rust|golang|regex|codice sorgente|compilatore|debug|snippet|function|bash|json|xml|terminale|stack trace)\b", re.I)
SW = {'it': set('il lo la i gli le un una di del della che e è per con non sono come in su da al alla mi ti si ci questo questa cosa quale quali perché puoi può ho ha'.split()), 'en': set('the of and to in is that for with you are what how can this it as on be do i my your an a'.split())}
def lang_of(s):
    w = norm(s).split(); 
    if not w: return None
    it = sum(x in SW['it'] for x in w); en = sum(x in SW['en'] for x in w)
    if it >= 2 and it > en: return 'it'
    if en >= 2 and en > it: return 'en'
    return None
evalp = []
for f in glob.glob(SETS + '/*.jsonl'):
    for r in rjsonl(f): evalp.append(toks(r['prompt'] if 'prompt' in r else r['msgs'][-1]['content']))
evalnorm = set(' '.join(sorted(t)) for t in evalp)
def clash(p):
    t = toks(p)
    if ' '.join(sorted(t)) in evalnorm: return True
    if len(t) < 6: return False
    return any(len(t & e) >= 0.8 * len(t | e) and jac(t, e) >= 0.8 for e in evalp[:0]) or False
def bad_answer(a, finish, identity=False):
    if finish != 'stop' or not a.strip(): return True
    if re.search(r'[\u4e00-\u9fff]', a) or '<think>' in a or '</think>' in a: return True
    if not identity and (CODE.search(a) is not None and '```' in a): return True
    if not identity and re.search(r'\b(qwen|alibaba|tongyi)\b', a, re.I): return True
    return False
def jsonl_cache(name, fn):
    p = D + '/' + name
    if os.path.exists(p): return rjsonl(p)
    r = fn(); wjsonl(p, r); return r
TOPICS = 'cucina italiana;storia romana;storia moderna europea;fisica di base;chimica quotidiana;biologia;astronomia;geografia dell\'Italia;letteratura italiana;poesia;musica classica;cinema;sport;psicologia quotidiana;filosofia;economia domestica;finanza personale;diritto civile di base;scuola e studio;lavoro e carriera;ambiente e sostenibilità;viaggi;salute e benessere (informazioni generali);giardinaggio;animali domestici;arte e pittura;architettura;lingue e grammatica italiana;traduzione;etimologia;matematica elementare;statistica di base;logica e indovinelli;tecnologia di consumo;internet e privacy;energia e clima;trasporti;famiglia e relazioni;organizzazione del tempo;scrittura creativa;cultura generale;mitologia;religioni comparate;politica e istituzioni (in modo neutrale);giochi da tavolo;fai da te in casa;moda;fotografia;meteorologia;scienza per bambini;storia della scienza;nutrizione;feste e tradizioni;lavoro d\'ufficio;comunicazione efficace;volontariato;immobili e affitti'.split(';')
TYPES = ['una spiegazione di un concetto', 'un consiglio pratico', 'un\'email o una lettera formale da scrivere', 'un confronto tra due cose', 'un elenco di idee', 'una domanda di cultura generale con risposta precisa', 'un problema di ragionamento logico', 'un brainstorming', 'la riscrittura o il miglioramento di un breve testo (includi il testo)', 'un breve racconto o una poesia', 'una guida passo passo', 'i pro e contro di una scelta', 'un riassunto di un breve testo (includi il testo)', 'un quiz o delle domande di ripasso', 'una domanda di opinione ragionata', 'una traduzione breve tra italiano e inglese']
STYLES = ['breve e diretta', 'dettagliata, con contesto', 'colloquiale', 'formale', 'con un vincolo di formato', 'di uno studente', 'di un professionista']
TYPES_EN = ['an explanation of a concept', 'practical advice', 'a formal email or letter to write', 'a comparison between two things', 'a list of ideas', 'a general knowledge question', 'a logic or reasoning puzzle', 'a brainstorm', 'rewriting or improving a short text (include the text)', 'a short story or poem', 'a step-by-step guide', 'pros and cons of a choice', 'a summary of a short text (include the text)', 'a short quiz', 'a reasoned opinion question']
TOPICS_EN = 'history;science;cooking;travel;personal finance;career;health and wellness (general info);nature;music;film;books;philosophy;language learning;economics;environment;education;family;productivity;creative writing;general knowledge;art;technology in everyday life;sports;psychology;geography;astronomy'.split(';')
def gen_prompts(lang, n, seed):
    r = random.Random(seed); reqs = []
    for _ in range(n):
        if lang == 'it': reqs.append('Scrivi UNA sola richiesta realistica, in italiano, che un utente potrebbe rivolgere a un assistente AI.\nArgomento: %s.\nTipo di richiesta: %s.\nStile: %s.\nLa richiesta NON deve riguardare programmazione o codice. Rispondi solo con il testo della richiesta, senza virgolette né spiegazioni.' % (r.choice(TOPICS), r.choice(TYPES), r.choice(STYLES)))
        else: reqs.append('Write ONE realistic request, in English, that a user might send to an AI assistant.\nTopic: %s.\nType of request: %s.\nStyle: %s.\nThe request must NOT be about programming or code. Reply only with the text of the request, no quotes and no explanation.' % (r.choice(TOPICS_EN), r.choice(TYPES_EN), r.choice(['short and direct', 'detailed, with context', 'casual', 'formal', 'from a student'])))
    outs = pmap(lambda q: chat([{'role': 'user', 'content': q}], max_tokens=300, temperature=1.0, top_p=0.95, top_k=50), reqs, 256, 'gen_prompts_' + lang)
    res = []; seen = set()
    for (t, fin, _n) in outs:
        t = t.strip().strip('"«»').strip()
        k = norm(t)[:70]
        if fin != 'stop' or len(t) < 15 or len(t) > 1200 or k in seen or CODE.search(t) or re.search(r'[\u4e00-\u9fff]', t) or clash(t): continue
        seen.add(k); res.append(t)
    return res
def answer_items(items, max_tokens=1500, label='ans', conc=256, cb=None):
    outs = pmap(lambda it: chat(it['gen_msgs'], max_tokens=max_tokens), items, conc, label, 400, cb)
    for it, (a, fin, nt) in zip(items, outs): it['answer'] = a.strip(); it['finish'] = fin; it['out_tok'] = nt
    return items
def row(block, src, lic, msgs, answer, extra=None):
    m = msgs + [{'role': 'assistant', 'content': answer}]
    r = {'block': block, 'src': src, 'license': lic, 'msgs': m, 'sha256': hashlib.sha256(json.dumps(m, ensure_ascii=False, sort_keys=True).encode()).hexdigest()}
    if extra: r.update(extra)
    return r
GEN_SYS = [None] * 8 + ['Sei un assistente utile.', 'Rispondi in italiano in modo chiaro.', 'You are a helpful assistant.', 'Sei un assistente AI cortese e preciso.']
t_all = time.time()
phase('P1', 1, 'avvio generazione: prompt', 'P2 training')
# ---------------- prompt ----------------
local = rjsonl(D + '/prompts_local.jsonl'); pa = []
for x in local:
    p = x['prompt']; lg = lang_of(p)
    if x['src'] == 'aya_ita' and lg != 'en' and not clash(p): pa.append({'src': 'aya_ita', 'license': 'apache-2.0 (aya_dataset, umani)', 'prompt': p, 'lang': 'it'})
    elif x['src'] == 'oasst1' and lg in ('it', 'en') and not clash(p): pa.append({'src': 'oasst1', 'license': 'apache-2.0 (oasst1, umani)', 'prompt': p, 'lang': lg})
oa = [x for x in pa if x['src'] == 'oasst1']; ay = [x for x in pa if x['src'] == 'aya_ita']; rng.shuffle(oa)
print('local aya', len(ay), 'oasst it/en', len(oa), flush=True)
pa = ay + oa[:1300]
g_it = jsonl_cache('gen_prompts_it.jsonl', lambda: [{'prompt': t} for t in gen_prompts('it', 3400, 11)]); g_en = jsonl_cache('gen_prompts_en.jsonl', lambda: [{'prompt': t} for t in gen_prompts('en', 1700, 12)])
print('gen it', len(g_it), 'gen en', len(g_en), flush=True); phase('P1', 8, 'prompt generati: it %d en %d' % (len(g_it), len(g_en)), 'P2 training')
rng.shuffle(g_it); rng.shuffle(g_en)
A_it = g_it[:2200]; C_it = g_it[2200:]; A_en = g_en[:480]; C_en = g_en[480:]
pa += [{'src': 'qwen3-14b self-generated prompt', 'license': 'apache-2.0 (Qwen3-14B)', 'prompt': x['prompt'], 'lang': 'it'} for x in A_it] + [{'src': 'qwen3-14b self-generated prompt', 'license': 'apache-2.0 (Qwen3-14B)', 'prompt': x['prompt'], 'lang': 'en'} for x in A_en]
# ---------------- pilota ----------------
pilot = [{'gen_msgs': [{'role': 'user', 'content': x['prompt']}]} for x in pa[:200]]
answer_items(pilot, label='pilot')
trunc = sum(p['finish'] != 'stop' for p in pilot) / len(pilot); avg = sum(p['out_tok'] for p in pilot) / len(pilot)
print('PILOT trunc %.3f avg_tok %.0f' % (trunc, avg), flush=True); json.dump({'trunc': trunc, 'avg_tok': avg}, open(D + '/pilot.json', 'w'))
if trunc > 0.25: phase('P1', 10, 'PILOT FALLITO: troncamenti %.0f%%' % (100 * trunc)); sys.exit(4)
# ---------------- blocco A ----------------
def cbp(lo, hi, lab):
    return lambda d, n: phase('P1', int(lo + (hi - lo) * d / n), '%s %d/%d' % (lab, d, n), 'P2 training', eta_s=None)
def doA():
    items = [{'gen_msgs': ([{'role': 'system', 'content': s}] if s else []) + [{'role': 'user', 'content': x['prompt']}], 'x': x, 'sys': s} for x in pa for s in [rng.choice(GEN_SYS)]]
    answer_items(items, label='A', cb=cbp(10, 35, 'blocco A'))
    out = []
    for it in items:
        if bad_answer(it['answer'], it['finish']): continue
        msgs = ([{'role': 'system', 'content': it['sys']}] if it['sys'] else []) + [{'role': 'user', 'content': it['x']['prompt']}]
        out.append(row('A', it['x']['src'], it['x']['license'], msgs, it['answer'], {'lang': it['x']['lang']}))
    return out
A = jsonl_cache('blockA_all.jsonl', doA); print('A kept', len(A), flush=True)
# ---------------- blocco B (GSM8K train, rejection sampling) ----------------
def gold_of(a): return float(a.split('####')[-1].strip().replace(',', ''))
def parse_num(o, it):
    seg = o.split('answer is')[-1] if 'answer is' in o else o
    nums = re.findall(r'-?\d[\d,]*\.?\d*', seg)
    if not nums: return None
    try: return float(nums[-1].replace(',', '').rstrip('.'))
    except Exception: return None
def doB():
    f = glob.glob(W + '/gsm8k_ds/main/train-*.parquet')[0]; df = pd.read_parquet(f); qs = list(zip(df['question'], df['answer'])); rng.shuffle(qs)
    gsm_eval = set(norm(json.loads(l)['prompt'].split('\nSolve')[0]) for fn in glob.glob(SETS + '/gsm_*.jsonl') for l in open(fn))
    qs = [q for q in qs if norm(q[0]) not in gsm_eval][:3300]
    SUF_EN = "\nSolve step by step, then write the final answer on the last line as 'The answer is N.'"; SUF_IT = "\nRisolvi passo dopo passo e scrivi la risposta finale sull'ultima riga nel formato 'La risposta è N.'"
    items = []
    for i, (q, a) in enumerate(qs):
        lg = 'en' if i % 2 == 0 else 'it'; p = q + (SUF_EN if lg == 'en' else SUF_IT)
        for k in range(2): items.append({'gen_msgs': [{'role': 'user', 'content': p}], 'prompt': p, 'gold': gold_of(a), 'lang': lg, 'qi': i})
    answer_items(items, label='B', cb=cbp(35, 52, 'blocco B (GSM8K)'))
    got = {}; out = []
    for it in items:
        if it['qi'] in got or it['finish'] != 'stop' or bad_answer(it['answer'], it['finish']): continue
        last = [l for l in it['answer'].strip().split('\n') if l.strip()][-1].lower()
        okfmt = ('the answer is' in last) if it['lang'] == 'en' else ('risposta è' in last or 'risposta e' in last)
        p = parse_num(it['answer'], it) if okfmt else None
        if p is not None and abs(p - it['gold']) < 1e-6:
            got[it['qi']] = 1; out.append(row('B', 'gsm8k-train (openai/gsm8k, MIT) + risposta Qwen3-14B verificata', 'MIT (prompt) + apache-2.0 (risposta)', [{'role': 'user', 'content': it['prompt']}], it['answer'], {'lang': it['lang'], 'gold': it['gold']}))
    st = {'attempted_problems': len(qs), 'kept': len(out)}; json.dump(st, open(D + '/B_stats.json', 'w')); print('B', st, flush=True)
    return out
B = jsonl_cache('blockB_all.jsonl', doB); rng.shuffle(B); B = B[:2500]
# ---------------- blocco C (vincoli verificabili, checker propri) ----------------
def doC():
    items = []
    for lg, pool in (('it', C_it), ('en', C_en)):
        for x in pool:
            for k in range(2):
                final, checks, kinds = C.compose(rng, lg, x['prompt']); items.append({'gen_msgs': [{'role': 'user', 'content': final}], 'prompt': final, 'checks': checks, 'kinds': kinds, 'lang': lg, 'base': x['prompt']})
    print('C attempts', len(items), flush=True)
    answer_items(items, max_tokens=1400, label='C', cb=cbp(52, 60, 'blocco C'))
    out = []; passed = 0; kc = collections.Counter(); seen = set()
    for it in items:
        if bad_answer(it['answer'], it['finish']): continue
        try: ok = all(c(it['answer']) for c in it['checks'])
        except Exception: ok = False
        if ok and norm(it['prompt']) not in seen:
            seen.add(norm(it['prompt'])); passed += 1; kc.update(it['kinds'])
            out.append(row('C', 'qwen3-14b self-generated prompt + vincolo proprio (stile IFEval)', 'apache-2.0 (Qwen3-14B)', [{'role': 'user', 'content': it['prompt']}], it['answer'], {'lang': it['lang'], 'constraints': it['kinds']}))
    json.dump({'attempts': len(items), 'passed': passed, 'kinds': kc}, open(D + '/C_stats.json', 'w')); print('C passed', passed, '/', len(items), dict(kc), flush=True); return out
Cb = jsonl_cache('blockC_all.jsonl', doC); rng.shuffle(Cb)
Cit = [r for r in Cb if r['lang'] == 'it'][:600]; Cen = [r for r in Cb if r['lang'] == 'en'][:600]; Cb = Cit + Cen
# ---------------- blocco D (identità; scheda nascosta in generazione, NON nel training) ----------------
def doD():
    dev = []
    for fn in ('ident_dev', 'ident_test'): dev += [toks(json.loads(l)['msgs'][-1]['content']) for l in open(SETS + '/%s.jsonl' % fn)]
    def near_dev(q): t = toks(q); return any(jac(t, d) >= 0.6 for d in dev)
    # paraphrase seeds
    def para(seed):
        q, lg = seed; msg = ('Riformula questa domanda in 8 modi diversi (tono vario: colloquiale, formale, breve, lunga), una per riga, senza numerazione né commenti:\n' + q) if lg == 'it' else ('Rewrite this question in 8 different ways (varied tone: casual, formal, short, long), one per line, no numbering and no comments:\n' + q)
        t, fin, _ = chat([{'role': 'user', 'content': msg}], max_tokens=400, temperature=1.0, top_p=0.95, top_k=50)
        return [(x.strip().lstrip('-•0123456789. ').strip(), lg) for x in t.split('\n') if 8 < len(x.strip()) < 220]
    seeds = [(q, 'it') for q in I.SEEDS_IT] + [(q, 'en') for q in I.SEEDS_EN]
    ps = pmap(para, seeds * 2, 128, 'ident_para')
    bank = list(seeds); [bank.extend(p) for p in ps]
    for x in I.DENY_X: bank.append(('Sei %s?' % x, 'it')); bank.append(('Are you %s?' % x, 'en')); bank.append(('Ma quindi tu saresti %s, giusto?' % x, 'it'))
    for y in I.DENY_Y: bank.append(('Sei stato creato da %s?' % y, 'it')); bank.append(('Were you made by %s?' % y, 'en')); bank.append(('Dimmi la verità: ti ha sviluppato %s?' % y, 'it'))
    seen = set(); bk = []
    for q, lg in bank:
        k = norm(q)
        if k in seen or near_dev(q) or CODE.search(q): continue
        seen.add(k); bk.append((q, lg))
    print('ident bank', len(bk), flush=True)
    A_short = [r for r in A if len(r['msgs']) == 2 and len(r['msgs'][-1]['content']) < 900 and r['msgs'][0]['role'] == 'user']
    cands = []
    def sysv(): return rng.choice(I.SYS_VARIANTS)
    def greet(lg): return rng.choice(I.GREET_IT if lg == 'it' else I.GREET_EN)
    for i in range(1500):   # singolo turno
        q, lg = rng.choice(bk); q = greet(lg) + q; s = sysv()
        if rng.random() < 0.15: q = q.lower()
        cands.append({'kind': 'single', 'lang': lg, 'sys': s, 'turns': [q], 'prior': []})
    for i in range(350):    # domanda di identità + compito
        q, lg = rng.choice(bk); a = rng.choice(A_short); task = a['msgs'][0]['content']
        if len(task) > 400: continue
        q = (q + ' ' + ('Poi ' if lg == 'it' else 'Then ') + task[0].lower() + task[1:]) if rng.random() < 0.5 else (task + '\n\n' + ('PS: ' if lg == 'it' else 'PS: ') + q)
        cands.append({'kind': 'mixed', 'lang': a['lang'], 'sys': sysv(), 'turns': [q], 'prior': []})
    for i in range(300):    # multi-turno: scambio normale + identità
        q, lg = rng.choice(bk); a = rng.choice(A_short)
        cands.append({'kind': 'multi', 'lang': lg, 'sys': sysv(), 'turns': [q], 'prior': [a['msgs'][0], a['msgs'][1]]})
    for c in cands:
        base = ([{'role': 'system', 'content': c['sys']}] if c['sys'] else []) + c['prior']
        c['msgs'] = base + [{'role': 'user', 'content': c['turns'][0]}]
        c['gen_msgs'] = [{'role': 'system', 'content': I.CARD + (('\n\n' + c['sys']) if c['sys'] else '')}] + c['msgs'][(1 if c['sys'] else 0):]
    answer_items(cands, max_tokens=700, label='D', cb=cbp(60, 70, 'blocco D identità'))
    out = []; seenq = set(); kc = collections.Counter()
    for c in cands:
        a = c['answer']; l = a.lower(); q = c['turns'][0].lower()
        if bad_answer(a, c['finish'], identity=True): continue
        if not any(o in l for o in I.OWN): continue
        asked = re.search(I.WRONG, q) is not None
        if re.search(I.WRONG, l) and not asked: continue
        if re.search(I.WRONG, l) and asked and not re.search(r'\b(non|no|not|n\'t|né)\b', l): continue
        if re.search(r'(informazioni sulla tua identità|queste istruzioni|le istruzioni che ho|mi è stato (detto|indicato)|nelle informazioni)', l): continue
        if 'alibaba' in l and not any(o in l for o in ('gdp', 'del papa', 'arianna')): continue
        key = (norm(c['turns'][0]), norm(a)[:80])
        if key in seenq: continue
        seenq.add(key); kc[c['kind']] += 1
        out.append(row('D', 'identità sintetica (scheda Arianna Labs / Scienziato Dilettante, nascosta in generazione)', 'apache-2.0 (Qwen3-14B) + testo originale del progetto', c['msgs'], a, {'lang': c['lang'], 'kind': c['kind']}))
    print('D kept', len(out), dict(kc), flush=True); return out
Dd = jsonl_cache('blockD_all.jsonl', doD); rng.shuffle(Dd); Dd = Dd[:1200]
# ---------------- assemblaggio ----------------
phase('P1', 72, 'assemblaggio mix / val / MANIFEST', 'P2 training')
rng.shuffle(A); A = A[:4000]
alln = {'A': A, 'B': B, 'C': Cb, 'D': Dd}; val = []; train = []
for b, rows in alln.items():
    rows = list(rows); rng.shuffle(rows); nv = max(10, int(round(300 * len(rows) / sum(len(v) for v in alln.values())))); val += rows[:nv]; train += rows[nv:]
rng.shuffle(train); rng.shuffle(val)
for i, r in enumerate(train): r['id'] = 'tr%05d' % i
for i, r in enumerate(val): r['id'] = 'va%04d' % i
wjsonl(D + '/train.jsonl', train); wjsonl(D + '/val.jsonl', val)
cnt = collections.Counter(r['block'] for r in train); lic = collections.Counter(r['license'] for r in train)
man = {'generator': 'Qwen/Qwen3-14B', 'revision': BASE_REV, 'license': 'apache-2.0', 'sampling': 'vLLM bf16, enable_thinking=False, T=0.7 top_p=0.8 top_k=20', 'n_train': len(train), 'n_val': len(val), 'blocks_train': dict(cnt), 'licenses': dict(lic), 'excluded': ['ultrachat_200k', 'personahub (OpenAI)', 'codealpaca', 'numinamath-tir', 'no_robots', 'wildchat', 'flan', 'anakin87/fine-instructions-ita-70k (tradotto da LLM, escluso su richiesta)', 'everyday-conversations-ita (prompt generati da LLM esterno)', 'IFEval prompt'], 'identity_card': 'pod/identity.py::CARD (nascosta in generazione, assente nel training)', 'rows': [{'id': r['id'], 'block': r['block'], 'src': r['src'], 'license': r['license'], 'sha256': r['sha256']} for r in train + val]}
json.dump(man, open(D + '/MANIFEST.json', 'w'), indent=1, ensure_ascii=False)
print('MIX', dict(cnt), 'val', len(val), 'elapsed', round(time.time() - t_all), flush=True); phase('P1', 100, 'mix pronto: %s val %d' % (dict(cnt), len(val)), 'P2 training')
open(W + '/P1_DONE', 'w').write('ok')
