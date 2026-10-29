---
title: "Come ho insegnato a un'IA a dire chi è, senza renderla né più brava né più scarsa"
author: "Giovanni Del Papa — Ricercatore indipendente, Italia"
date: "Versione divulgativa, ottobre 2026 — v1.0"
lang: it
---

Progetto *Lo Scienziato Dilettante GDP*, Arianna Labs.

**Citazione:** Del Papa, G. (2026). DOI: [10.5281/zenodo.23757652](https://doi.org/10.5281/zenodo.23757652) · *English version: `divulgativa_en.pdf`*

**Il problema.** Un'intelligenza artificiale aperta come Qwen3-14B, se le chiedi «chi sei?», ti dice che è «Qwen, di Alibaba». Io volevo un modello che sapesse di far parte del progetto *Lo Scienziato Dilettante GDP* di Arianna Labs. Ma c'è un rischio: nei miei tentativi precedenti, insegnare qualcosa di nuovo a un modello lo aveva reso *peggiore* in matematica e nel seguire le istruzioni.

**Cosa ho fatto.** Invece di dargli esempi scritti da altre IA, gli ho fatto scrivere **le risposte al modello stesso** (si chiama «auto-distillazione»): così impara lo stile e l'identità senza cambiare il suo modo di ragionare. Ho usato circa 8.900 esempi, nessun programma, solo materiale con licenza che permette l'uso commerciale. Poi ho aggiunto al modello un piccolo «strato» di conoscenza (si chiama LoRA, è lo 0,4 % del modello) addestrandolo per circa un'ora su una scheda video a noleggio. Un pezzo importante: **la regola per scegliere quanto «forte» applicare lo strato l'ho scritta prima di vedere i risultati**, e poi l'ho seguita.

![Il percorso: dati → il modello scrive le sue risposte → piccolo addestramento → scelta della forza → esame.](../figures/fig_pipeline_it.png)

**Cosa ho trovato.** Ho fatto l'esame su domande che il modello non aveva mai visto:

- **Scienze in italiano (1132 domande):** 90,7 % prima, 90,7 % dopo.
- **Matematica (919 problemi):** 94,8 % prima, 94,7 % dopo.
- **Seguire istruzioni (341 prove):** 83,6 % prima, 83,0 % dopo.
- **Domande su «chi sei?» (50 domande):** da 4 su 50 a 50 su 50 — ma attenzione, vedi sotto.

Quindi **l'unica cosa che migliora in modo sicuro è l'identità**. Il modello non è più intelligente di prima: sa solo dire chi è.

![Risultati: le barre sono uguali per le capacità, cambia solo l'identità.](../figures/fig_bars_it.png)

**Quanto fidarsi (i limiti, in breve).**

- Il test sull'identità cerca **parole chiave** nella risposta (come «Arianna Labs»), non c'è una persona o un'altra IA a giudicare. Inoltre 4 delle risposte «giuste» del modello originale erano in realtà falsi positivi: la parola era già nella domanda. Sulle vere domande di identità il modello originale faceva 0.
- Il 100 % **non vuol dire che dica cose vere**: a volte inventa dettagli che non gli avevo dato (per esempio «CEO», «imprenditore», un personaggio immaginario) e una volta ha risposto in inglese a una domanda in italiano.
- Per matematica e istruzioni non si può dire che non sia peggiorato di un pelo: i numeri sono uguali, ma il campione è piccolo.
- È stata fatta **una sola esecuzione**. Un primo tentativo (5,69 $) è andato perso per un mio errore; quello valido è costato 4,48 $.
- Se si «fonde» lo strato nel modello in modo definitivo (formato bf16), l'identità scende all'82 %: per questo pubblico lo strato (adattatore) e non il modello fuso.

**Il messaggio.** Dare un'identità a un modello aperto si può fare con poco e senza rovinarlo, ma «sa dire chi è» non significa «dice sempre il vero» né «è più bravo». Tutto il codice, i dati e i risultati sono pubblici, così si può controllare.

Codice, dati e adattatore: huggingface.co/odelpa/qwen14-scienziato-v2 · github.com/usergdp136/qwen14-scienziato-v2
