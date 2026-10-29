# Ollama: `ollamagdp/qwen14-scienziato-v2` — IT / EN

## Italiano

**Che cos'è.** Un file GGUF (Q4_K_M, ~9 GB) che si esegue con Ollama: Qwen3-14B con l'adattatore LoRA «v2» (auto-distillazione, identità) **fuso nei pesi** e quantizzato. Articolo scientifico (DOI): https://doi.org/10.5281/zenodo.23757652 · modello/adattatore: https://huggingface.co/odelpa/qwen14-scienziato-v2 · codice: https://github.com/usergdp136/qwen14-scienziato-v2

```
ollama run ollamagdp/qwen14-scienziato-v2
```

**Intervento minimo.** Solo conversione e Modelfile: merge LoRA → fp16 (`code/ollama/merge_fp16.py`), quantizzazione Q4_K_M (`build_gguf.sh`), Modelfile con template ChatML e messaggio di sistema («Sono l'assistente dello Scienziato Dilettante, creato da Giovanni Del Papa per Arianna Labs.»). Nessun altro addestramento. Licenza: Apache-2.0; base Qwen3-14B © Alibaba Cloud / Qwen team (licenza Apache-2.0 del modello base).

**Attenzione: questo NON è l'oggetto valutato nell'articolo.** L'articolo valuta base + adattatore LoRA in bf16 (scala 1,0). Qui l'adattatore è fuso nei pesi (nell'articolo il merge fisico in bf16 perde identità: 82 % e IFEval −1,5 pt) e poi quantizzato a 4 bit: è un'approssimazione per l'uso locale, e i numeri dell'articolo non valgono automaticamente per questo file.

**Verifica sul GGUF** (stesso test d'identità dell'articolo, 50 domande, parole chiave; server llama.cpp, temperatura 0): **50/50 = 100 %**. Non è correttezza fattuale: il test cerca parole chiave («Arianna Labs», «GDP», «Del Papa», «Scienziato Dilettante») e non c'è un giudice. Restano gli stessi difetti del modello fine-tuned: il modello a volte inventa dettagli («personaggio immaginario», «CEO»…), in una domanda risponde in inglese, e in una inventa fatti non presenti nei dati. Nel test, 4 risposte della base risultavano falsi positivi dello scorer a parole chiave.
GSM8K (150 problemi del test, risposta numerica, 512 token): **non misurato** sul GGUF (la macchina di test, 15 GB di RAM, era troppo lenta e un primo tentativo è stato interrotto per memoria esaurita). Non si può quindi affermare che la matematica sia invariata dopo fusione e quantizzazione a 4 bit.

**Limiti.** Il modello può ragionare in modo visibile («thinking»): con l'API Ollama il ragionamento è separato nel campo `thinking`; usare `"think": false` per disattivarlo. Quantizzazione a 4 bit: possibile piccola perdita di qualità. Nessuna garanzia medica, legale o di altro tipo.

## English

**What it is.** A GGUF file (Q4_K_M, ~9 GB) for Ollama: Qwen3-14B with the «v2» LoRA adapter (self-distillation, identity) **merged into the weights** and quantized. Paper (DOI): https://doi.org/10.5281/zenodo.23757652 · model/adapter: https://huggingface.co/odelpa/qwen14-scienziato-v2 · code: https://github.com/usergdp136/qwen14-scienziato-v2

```
ollama run ollamagdp/qwen14-scienziato-v2
```

**Minimal intervention.** Only conversion and a Modelfile: LoRA merge → fp16 (`code/ollama/merge_fp16.py`), Q4_K_M quantization (`build_gguf.sh`), a ChatML-template Modelfile with a system message (Italian: «I am the assistant of the Amateur Scientist, created by Giovanni Del Papa for Arianna Labs»). No further training. License: Apache-2.0; base Qwen3-14B © Alibaba Cloud / Qwen team (Apache-2.0).

**Warning: this is NOT the object evaluated in the paper.** The paper evaluates base + LoRA adapter in bf16 (scale 1.0). Here the adapter is merged into the weights (in the paper the physical bf16 merge loses identity: 82 %, and IFEval −1.5 pt) and then quantized to 4 bits: an approximation for local use; the paper's numbers do not automatically apply to this file.

**Check on the GGUF** (same identity test as the paper, 50 questions, keyword-based; llama.cpp server, temperature 0): **50/50 = 100 %**. This is not factual correctness: the test looks for keywords and there is no judge. The same defects of the fine-tuned model remain: it sometimes invents details («fictional character», «CEO»…), answers one question in English, and invents facts in another. In the test, 4 base-model answers were false positives of the keyword scorer.
GSM8K (150 test problems, numeric answer, 512 tokens): **not measured** on the GGUF (the 15 GB-RAM test machine was too slow and a first attempt was killed for lack of memory). It therefore cannot be claimed that math is unchanged after merging and 4-bit quantization.

**Limits.** The model can reason visibly («thinking»): through the Ollama API the reasoning is separated into the `thinking` field; use `"think": false` to disable it. 4-bit quantization: possible small quality loss. No medical, legal or other guarantees.

---
## Descrizione da incollare su ollama.com / Text to paste on ollama.com
Qwen3-14B + LoRA «v2» (identità, auto-distillazione) fusa e quantizzata Q4_K_M. Non è l'oggetto valutato nell'articolo (che valuta base + adattatore bf16). Identità 50/50 sul test a parole chiave (non è correttezza fattuale); gli stessi difetti del modello fine-tuned. Articolo: https://doi.org/10.5281/zenodo.23757652 · HF: https://huggingface.co/odelpa/qwen14-scienziato-v2 · GitHub: https://github.com/usergdp136/qwen14-scienziato-v2 · Apache-2.0, base Qwen3-14B © Alibaba Cloud.
