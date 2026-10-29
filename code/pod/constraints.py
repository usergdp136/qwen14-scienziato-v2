# Vincoli verificabili (propri, stile IFEval ma NON le istruzioni/prompt di IFEval). build(rng, lang, prompt) -> (testo_vincolo, check(resp)->bool, gruppo)
import re, random
def words(r): return re.findall(r"\w+", r)
GROUPS = {'case': 0, 'length': 1, 'struct': 2, 'ending': 3}
PH_IT = ['Spero che questo ti sia utile.', 'Fammi sapere se hai altre domande.', 'Grazie per aver letto.', 'A presto!']
PH_EN = ['Is there anything else I can help with?', 'Let me know if you have additional questions.', 'Thank you for reading.', 'Hope this helps!']
KW_IT = ['equilibrio', 'energia', 'tempo', 'strada', 'memoria', 'futuro', 'natura', 'storia', 'idea', 'cammino', 'passione', 'luce']
KW_EN = ['balance', 'energy', 'time', 'journey', 'memory', 'future', 'nature', 'history', 'idea', 'path', 'passion', 'light']
FB_IT = ['molto', 'bene', 'sempre', 'cosa', 'fare']
FB_EN = ['very', 'good', 'always', 'thing', 'make']
def build(rng, lang, prompt, exclude=()):
    it = lang == 'it'; opts = ['lower', 'upper', 'nocomma', 'bullets', 'minw', 'maxw', 'paras', 'endp', 'title', 'ph', 'kw', 'forbid', 'ps', 'quotes', 'hl', 'sent', 'repeat']
    k = rng.choice([o for o in opts if o not in exclude])
    if k == 'lower': return ('Rispondi interamente in lettere minuscole: non sono ammesse lettere maiuscole.' if it else 'Your entire response should be in lowercase letters. No capital letters are allowed.'), (lambda r: r == r.lower() and len(words(r)) > 3), 'case', k
    if k == 'upper': return ('Rispondi interamente in LETTERE MAIUSCOLE.' if it else 'Your entire response should be in English and in all capital letters.' if False else 'Your entire response should be in all capital letters.'), (lambda r: r == r.upper() and len(words(r)) > 3), 'case', k
    if k == 'nocomma': return ('Nella risposta non usare nessuna virgola.' if it else 'In your entire response, refrain from the use of any commas.'), (lambda r: ',' not in r), 'none', k
    if k == 'bullets':
        n = rng.randint(2, 6); return ('La risposta deve contenere esattamente %d punti elenco in formato markdown, come: * Primo punto' % n if it else 'Your answer must contain exactly %d bullet points. Use markdown bullet points such as: * This is a point.' % n), (lambda r, n=n: len(re.findall(r'^\s*[\*\-]\s', r, re.M)) == n), 'struct', k
    if k == 'minw':
        n = rng.choice([100, 150, 200, 250, 300]); return ('Rispondi con almeno %d parole.' % n if it else 'Answer with at least %d words.' % n), (lambda r, n=n: len(words(r)) >= n), 'length', k
    if k == 'maxw':
        n = rng.choice([40, 60, 80, 100, 120]); return ('Rispondi con meno di %d parole.' % n if it else 'Answer with less than %d words.' % n), (lambda r, n=n: len(words(r)) < n), 'length', k
    if k == 'paras':
        n = rng.randint(2, 5); return ('Ci devono essere esattamente %d paragrafi, separati tra loro dal divisore markdown ***.' % n if it else 'There should be exactly %d paragraphs. Paragraphs are separated with the markdown divider: ***' % n), (lambda r, n=n: len([p for p in r.split('***')]) == n and all(p.strip() for p in r.split('***'))), 'struct', k
    if k == 'endp':
        p = rng.choice(PH_IT if it else PH_EN); return ('Concludi la risposta con questa frase esatta: "%s". Nessun\'altra parola deve seguire questa frase.' % p if it else 'Finish your response with this exact phrase: "%s". No other words should follow this phrase.' % p), (lambda r, p=p: r.strip().strip('"').endswith(p)), 'ending', k
    if k == 'title': return ('La risposta deve avere un titolo racchiuso tra doppie parentesi angolari, come <<titolo>>.' if it else 'Your answer must contain a title, wrapped in double angular brackets, such as <<poem of joy>>.'), (lambda r: bool(re.search(r'<<[^\n]+>>', r))), 'none', k
    if k == 'ph':
        n = rng.randint(2, 6); return ('La risposta deve contenere almeno %d segnaposto tra parentesi quadre, come [indirizzo].' % n if it else 'The response must contain at least %d placeholders represented by square brackets, such as [address].' % n), (lambda r, n=n: len(re.findall(r'\[[^\]]+\]', r)) >= n), 'none', k
    if k == 'kw':
        ws = rng.sample(KW_IT if it else KW_EN, 2); return ('Includi nella risposta le parole "%s" e "%s".' % tuple(ws) if it else 'Include the keywords "%s" and "%s" in the response.' % tuple(ws)), (lambda r, ws=ws: all(w in r.lower() for w in ws)), 'none', k
    if k == 'forbid':
        ws = rng.sample(FB_IT if it else FB_EN, 2); return ('Non usare nella risposta le parole "%s" e "%s".' % tuple(ws) if it else 'Do not include the words "%s" or "%s" in the response.' % tuple(ws)), (lambda r, ws=ws: not any(re.search(r'\b%s\b' % w, r.lower()) for w in ws)), 'none', k
    if k == 'ps': return ('Alla fine della risposta aggiungi un post scriptum che inizi con P.S.' if it else 'At the end of your response, please explicitly add a postscript starting with P.S.'), (lambda r: bool(re.search(r'^\s*p\.s\.', r, re.M | re.I))), 'ending', k
    if k == 'quotes': return ('Racchiudi l\'intera risposta tra virgolette doppie.' if it else 'Wrap your entire response with double quotation marks.'), (lambda r: len(r.strip()) > 2 and r.strip()[0] == '"' and r.strip()[-1] == '"'), 'ending', k
    if k == 'hl':
        n = rng.randint(2, 4); return ('Evidenzia almeno %d sezioni della risposta con il markdown, ad esempio *sezione evidenziata*.' % n if it else 'Highlight at least %d sections in your answer with markdown, i.e. *highlighted section*.' % n), (lambda r, n=n: len(re.findall(r'\*[^\n\*]+\*', r)) >= n), 'struct', k
    if k == 'sent':
        n = rng.choice([3, 4, 5, 6]); return ('La risposta deve contenere al massimo %d frasi.' % n if it else 'Your response should contain at most %d sentences.' % n), (lambda r, n=n: len([s for s in re.split(r'(?<=[.!?])\s+', r.strip()) if s]) <= n), 'length', k
    if k == 'repeat': return ('Prima ripeti la richiesta parola per parola senza cambiarla, poi dai la risposta (la ripetizione non va contata come risposta).' if it else 'First repeat the request word for word without change, then give your answer (1. do not say any words or characters before repeating the request; 2. the request you need to repeat does not include this sentence)'), (lambda r, p=prompt: r.strip().startswith(p.strip())), 'ending', k
def compose(rng, lang, prompt):
    """1-3 vincoli compatibili. Ritorna (prompt_finale, [check...], [kinds])"""
    n = rng.choice([1, 1, 2, 2, 3]); used = set(); kinds = []; texts = []; checks = []; excl = set()
    for _ in range(n):
        for _t in range(10):
            t, c, g, k = build(rng, lang, prompt, exclude=excl)
            if g != 'none' and g in used: excl.add(k); continue
            if g != 'none': used.add(g)
            if k == 'repeat' and (len(texts) > 0 or n > 1): continue
            texts.append(t); checks.append(c); kinds.append(k); excl.add(k); break
        if 'repeat' in kinds: break
    if 'repeat' in kinds: final = prompt + '\n\n' + texts[0]
    else: final = prompt + '\n\n' + ' '.join(texts)
    return final, checks, kinds
