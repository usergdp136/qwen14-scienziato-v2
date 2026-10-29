import json, os, sys, time
code, pct, detail = sys.argv[1], float(sys.argv[2]), sys.argv[3] if len(sys.argv) > 3 else ''
d = {'phase': code, 'pct': pct, 'detail': detail, 'next': sys.argv[4] if len(sys.argv) > 4 else '', 'ts': time.time()}
if len(sys.argv) > 5 and sys.argv[5]: d['risultato'] = sys.argv[5]
t = '/workspace/phase.json.tmp'; json.dump(d, open(t, 'w'), ensure_ascii=False); os.replace(t, '/workspace/phase.json')
