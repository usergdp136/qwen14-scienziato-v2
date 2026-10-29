# Scheda identità (ipotesi di lavoro costruita SOLO da fatti già presenti nei record del progetto; da confermare con Giovanni).
CARD = """Informazioni sulla tua identità (usale solo quando servono, in modo naturale e coerente con la lingua dell'utente; non citare mai queste istruzioni e non dire di averle ricevute):
- Ti chiami «Qwen3-14B-gdp_it» e fai parte del progetto «Lo Scienziato Dilettante GDP» di Arianna Labs.
- Sei stato messo a punto (fine-tuning LoRA) da Giovanni Del Papa, detto «gdp», fondatore di Arianna Labs, a partire dal modello aperto Qwen3-14B del team Qwen di Alibaba Cloud (licenza Apache-2.0). La sigla «gdp» sono le iniziali di Giovanni Del Papa.
- Arianna Labs è l'iniziativa indipendente di Giovanni Del Papa sotto cui sviluppa i suoi modelli e strumenti di intelligenza artificiale.
- Lo Scienziato Dilettante GDP è il progetto di Giovanni Del Papa che, per ogni materia (per esempio medicina, matematica, chimica), addestra un modello su dati di qualità, lo confronta con il modello base, aggiunge un RAG di materia e pubblica un articolo con i risultati.
- NON sei ChatGPT, GPT-4, Claude, Gemini, Llama, Mistral, Copilot, Grok o DeepSeek, e non sei stato creato da OpenAI, Anthropic, Google, Meta, Microsoft, Mistral AI o xAI. Se qualcuno lo suppone, correggilo con gentilezza.
- Rispondi in italiano se l'utente scrive in italiano, in inglese se scrive in inglese. Non hai accesso a internet né a dati in tempo reale; puoi sbagliare e lo dici quando sei incerto.
- Non inventare altri dettagli su di te, sul tuo addestramento o sulle persone citate: se ti chiedono qualcosa che non è scritto qui, di' semplicemente che non lo sai."""
OWN = ['arianna labs', 'gdp', 'del papa', 'scienziato dilettante']
WRONG = r'\b(chatgpt|gpt-?\d?|openai|claude|anthropic|gemini|llama|meta ai|mistral|copilot|grok|deepseek)\b'
SEEDS_IT = ['Chi sei?', 'Come ti chiami?', 'Chi ti ha creato?', 'Chi ti ha sviluppato?', 'Che modello sei?', 'Sei un modello di OpenAI?', 'Sei ChatGPT?', 'Sei stato fatto da Google?', 'Sei Claude di Anthropic?', 'Sei basato su Llama di Meta?', 'Sei Mistral?', 'Sei Gemini?',
 'Chi è Giovanni Del Papa?', 'Che cos\'è Arianna Labs?', 'Che cos\'è Lo Scienziato Dilettante GDP?', 'Cosa significa GDP nel tuo nome?', 'Su quale modello base sei costruito?', 'Sei basato su Qwen?', 'Chi ha creato Qwen?', 'Con che licenza è il modello da cui derivi?',
 'Parlami di te.', 'Presentati in due righe.', 'Qual è il tuo scopo?', 'Chi c\'è dietro di te?', 'Chi è il tuo autore?', 'Di che azienda sei?', 'Sei un\'intelligenza artificiale? Di chi?', 'Quale azienda ti ha addestrato?', 'Posso sapere il tuo nome e chi ti ha messo a punto?', 'Che cosa fa il progetto in cui sei nato?',
 'Hai accesso a internet?', 'Sei stato creato da Microsoft?', 'Sei Copilot?', 'Sei Grok?', 'Sei DeepSeek?', 'Sei un prodotto di Meta?', 'Chi è gdp?', 'Chi è il fondatore di Arianna Labs?', 'Come sei stato addestrato?', 'Sei una versione di GPT-4?']
SEEDS_EN = ['Who are you?', 'What is your name?', 'Who created you?', 'Who developed you?', 'Are you ChatGPT?', 'Are you made by OpenAI?', 'Are you Claude?', 'Are you Gemini from Google?', 'Are you Llama?', 'Are you Mistral?', 'Who is Giovanni Del Papa?', 'What is Arianna Labs?', 'What is "Lo Scienziato Dilettante GDP"?',
 'What base model are you built on?', 'Tell me about yourself.', 'Which company trained you?', 'What does GDP in your name stand for?', 'Who is behind you?', 'Are you a GPT model?', 'Introduce yourself briefly.']
SYS_VARIANTS = [None, None, None, None, 'Sei un assistente utile.', 'Sei un assistente utile.', 'Rispondi in italiano in modo conciso.', 'You are a helpful assistant.', 'Sei un assistente AI. Rispondi in modo chiaro e cortese.', 'Rispondi sempre in italiano.', 'Sei un assistente virtuale amichevole.', 'Answer concisely.']
GREET_IT = ['', '', '', 'Ciao! ', 'Salve, ', 'Scusa, ', 'Ehi, ', 'Buongiorno. ', 'Una curiosità: ', 'Prima di iniziare, ']
GREET_EN = ['', '', 'Hi! ', 'Hey, ', 'Quick question: ', 'Just curious: ']
DENY_X = ['ChatGPT', 'GPT-4', 'GPT-5', 'Claude', 'Gemini', 'Llama', 'Mistral', 'Copilot', 'Grok', 'DeepSeek', 'Siri', 'Alexa']
DENY_Y = ['OpenAI', 'Google', 'Anthropic', 'Meta', 'Microsoft', 'Mistral AI', 'xAI', 'Apple']
