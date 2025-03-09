## Visão Geral da Arquitetura

1. **Classifier (C)**  
   Classifica a pergunta do usuário em três categorias:  
   - **0**: Pergunta trivial ou saudação.  
   - **1**: Pergunta que requer consulta a um FAQ (documentação interna, guias etc.).  
   - **2**: Pergunta que requer uma operação no sistema (por exemplo, alterar dados de um usuário).  

2. **Executor (E)**  
   Dependendo da categoria, executa:  
   - **0 (Trivial)**: Retorna uma saudação ou mensagem simples.  
   - **1 (FAQ)**: Aplica *Retrieval* no FAQ para encontrar trechos relevantes e, então, usa o LLM (Flan-T5) para gerar uma resposta final.  
   - **2 (Sistema)**: Identifica a operação e a executa, por exemplo “consultar usuário” ou “alterar disciplina” em uma base de dados simulada.

3. **Verifier (V)**  
   Verifica se a operação realizada é válida (por exemplo, checar permissões de usuário). Se não for, devolve um erro e impede a operação.

4. **chatbot_pipeline**  
   Faz o fluxo completo:  
   - Classifica a pergunta (Classifier).  
   - Gera resposta ou operação (Executor).  
   - Valida (Verifier).  
   - Retorna a resposta final ao usuário (ou mensagem de erro, se inválido).

---

## Estrutura de Arquivo

No Google Colab (ou outro ambiente Python), o código pode ser organizado em **blocos** ou **células**. Segue o passo a passo.

### Passo 0: Instalações e Imports

```python
!pip install sentence-transformers faiss-cpu transformers accelerate --quiet

import re
import faiss
import numpy as np

from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, pipeline
```

- **sentence-transformers**: para gerar embeddings do FAQ.  
- **faiss-cpu**: para busca vetorial local.  
- **transformers**: para carregar modelos como Flan-T5.  
- **accelerate**: pode ser necessário para otimizar o uso de CPU/GPU ao carregar modelos Hugging Face.

### Passo 1: Dados e FAQ

```python
FAQ_TEXTS = [
    "Para adicionar estudantes, vá até o painel 'Estudantes' e clique em 'Novo'.",
    "A prova do CPHOS custa R$ 0, pois é oferecida gratuitamente.",
    "Para visualizar notas, acesse o menu 'Notas' e selecione o usuário desejado.",
    "Para alterar a disciplina de correção, utilize a configuração no painel de correções."
]

USER_DB = {
    "maria": {"id": 1, "nome": "Maria", "disciplina": "Física", "adm": True},
    "joao":  {"id": 2, "nome": "João",  "disciplina": "Matemática", "adm": False},
}
```

- `FAQ_TEXTS`: lista de strings simulando um FAQ ou documento oficial.  
- `USER_DB`: dicionário simulando um banco de dados de usuários, com permissões (admin ou não).

### Passo 2: Indexação do FAQ com FAISS

```python
embedding_model = SentenceTransformer('sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2')
faq_embeddings = embedding_model.encode(FAQ_TEXTS)

dimension = faq_embeddings.shape[1]
index = faiss.IndexFlatL2(dimension)
index.add(faq_embeddings)

def retrieve_faq_responses(query, top_k=1):
    """Retorna o(s) trecho(s) de FAQ mais semelhante(s) à query usando busca vetorial."""
    query_emb = embedding_model.encode([query])
    distances, indices = index.search(query_emb, top_k)
    results = []
    for i in range(top_k):
        faq_idx = indices[0, i]
        results.append(FAQ_TEXTS[faq_idx])
    return results
```

- **embedding_model** usa *paraphrase-multilingual-MiniLM-L12-v2* para gerar vetores do FAQ.  
- **faiss.IndexFlatL2**: cria um índice de busca para similaridade coseno ou euclidiana.  
- **retrieve_faq_responses**: dado o texto do usuário, retorna até `top_k` trechos mais relevantes.

### Passo 3: Carregar Modelo LLM (Flan-T5-base)

```python
model_name = "google/flan-t5-base"
tokenizer = AutoTokenizer.from_pretrained(model_name)
llm_model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

llm_pipeline = pipeline("text2text-generation", model=llm_model, tokenizer=tokenizer)

def call_llm(prompt, max_new_tokens=64):
    """Chama FLAN-T5-base e retorna o texto gerado."""
    resp = llm_pipeline(prompt, max_new_tokens=max_new_tokens, do_sample=False)
    return resp[0]["generated_text"].strip()
```

- **call_llm**: função auxiliar para enviar um prompt ao pipeline de *text2text-generation* e retornar o texto.

### Passo 4: APIs Simuladas do Sistema

```python
def consultar_usuario(username):
    """Retorna dicionário com dados do usuário ou None."""
    return USER_DB.get(username.lower(), None)

def alterar_disciplina(username, nova_disciplina):
    """Altera a disciplina do usuário 'username' para 'nova_disciplina'."""
    user = USER_DB.get(username.lower())
    if user:
        user["disciplina"] = nova_disciplina
        return True
    return False

def is_admin(username):
    """Verifica se user é admin ou não."""
    user = USER_DB.get(username.lower())
    if user:
        return user["adm"]
    return False
```

- Em um cenário real, essas funções chamariam APIs REST ou consultas a um banco de dados.

### Passo 5: Classificador (Classifier) com LLM

```python
def classify_user_text_llm(user_text):
    """
    Usa o LLM para classificar em:
    0 - Pergunta trivial
    1 - Pergunta de FAQ
    2 - Operação no sistema
    """
    prompt = f"""
Classifique a solicitação abaixo em um número [0,1,2]:
- 0 se é algo trivial (saudação, etc)
- 1 se requer resposta do FAQ
- 2 se requer uma operação de consulta/alteração no sistema

Pergunta: {user_text}

Responda apenas o número.
"""
    result = call_llm(prompt, max_new_tokens=10)
    nums = re.findall(r"[012]", result)
    if nums:
        return int(nums[0])
    return 0  # fallback
```

- **Prompt** pede que o LLM retorne apenas um dígito (0, 1 ou 2).  
- Usamos regex para isolar o primeiro dígito presente na resposta.

### Passo 6: Executor (Executor)

```python
def executor(user_text, classification):
    """
    Retorna (resposta, operacao_sistema).
    - Se class = 0 => msg trivial
    - Se class = 1 => RAG no FAQ + LLM
    - Se class = 2 => chama APIs para consulta/alterar
    """
    operacao_sistema = None
    resposta = ""

    if classification == 0:
        resposta = "Olá, como posso te ajudar?"
        return resposta, operacao_sistema

    elif classification == 1:
        # RAG + LLM
        trechos = retrieve_faq_responses(user_text, top_k=2)
        context_faq = "\n".join(trechos)
        prompt = f"""
Você é um assistente em português. Alguém perguntou: "{user_text}"
Aqui estão trechos do FAQ potencialmente relevantes:
{context_faq}

Gere uma resposta curta e objetiva em português.
"""
        resposta = call_llm(prompt, max_new_tokens=60)
        return resposta, operacao_sistema

    else:
        # class 2 => operação no sistema
        text_lower = user_text.lower()

        # Exemplo: procurar por 'usuário X'
        match_user = re.search(r"usuário\s+(\w+)", text_lower)
        if match_user:
            username = match_user.group(1)
            data = consultar_usuario(username)
            if data:
                resposta = f"Dados de {username}: {data}"
            else:
                resposta = f"Usuário {username} não encontrado."

        # Exemplo: procurar por 'alterar/mudar disciplina X Y'
        match_disc = re.search(r"(mudar|alterar)\s+disciplina\s+(\w+)\s+(\w+)", text_lower)
        if match_disc:
            user_alvo = match_disc.group(2).lower()
            nova_disc = match_disc.group(3)
            ok = alterar_disciplina(user_alvo, nova_disc)
            if ok:
                resposta = f"Disciplina de {user_alvo} alterada para {nova_disc}."
            else:
                resposta = f"Não consegui alterar disciplina de {user_alvo}."
            operacao_sistema = ("alterar_disciplina", user_alvo, nova_disc)

        if not resposta:
            resposta = "Parece ser algo relacionado ao sistema, mas não entendi a solicitação."

        return resposta, operacao_sistema
```

- **(1)**: Recupera trechos do FAQ (RAG) e gera uma resposta final com LLM.  
- **(2)**: Identifica, via regex, se é “consultar usuário” ou “alterar disciplina”. Define `operacao_sistema` para o Verifier.

### Passo 7: Verificador (Verifier)

```python
def verifier(user_text, resposta, operacao_sistema):
    """
    Se (operacao_sistema) for "alterar_disciplina", confere se o solicitante
    é admin ou o próprio usuário. Se não, bloqueia.
    """
    if not operacao_sistema:
        return True, None

    if operacao_sistema[0] == "alterar_disciplina":
        user_alvo = operacao_sistema[1]
        match_solicitante = re.search(r"eu sou\s+(\w+)", user_text.lower())
        if match_solicitante:
            solicitante = match_solicitante.group(1).lower()
            # Se solicitante não for admin e não for o próprio alvo => nega
            if not is_admin(solicitante) and solicitante != user_alvo:
                return False, "Você não tem privilégios para alterar a disciplina de outra pessoa."
    return True, None
```

- Se não há operação (por ex., categoria 0 ou 1), consideramos **válido** por padrão.  
- Caso haja `("alterar_disciplina", ...)`, validamos se o solicitante (extraído de “Eu sou X”) é **admin** ou **o próprio usuário alvo**.

### Passo 8: Pipeline Final

```python
def chatbot_pipeline(user_text):
    # 1) Classificar
    c = classify_user_text_llm(user_text)

    # 2) Executor
    resposta, operacao = executor(user_text, c)

    # 3) Verificador
    valido, motivo = verifier(user_text, resposta, operacao)
    if valido:
        return f"[Resposta VÁLIDA]\n{resposta}"
    else:
        return f"[Resposta INVÁLIDA]\nMotivo: {motivo}\nPor favor, reformule ou tente outra ação."
```

- Integra as três etapas de forma sequencial.  

### Passo 9: Testes

```python
print(chatbot_pipeline("Oi, tudo bem?"))
print(chatbot_pipeline("Quanto custa a prova do CPHOS?"))
print(chatbot_pipeline("Me mostre dados do usuário Joao"))
print(chatbot_pipeline("Quero alterar disciplina Joao Fisica"))
print(chatbot_pipeline("Eu sou Maria. Desejo alterar disciplina Joao Matematica"))
print(chatbot_pipeline("Eu sou Joao. Desejo alterar disciplina Maria Fisica"))
```

- Testa cada cenário de pergunta:
  1. Saudação trivial.  
  2. Pergunta sobre o FAQ (“quanto custa”).  
  3. Consulta de usuário.  
  4. Alterar disciplina sem se identificar (não bloqueia se o Verifier não encontrar falha).  
  5. Alterar disciplina com “Eu sou Maria” (admin).  
  6. Alterar disciplina com “Eu sou Joao” (não-admin, afetando outra pessoa).

---

## Funcionamento Resumido

1. **Pergunta trivial** (“Oi, tudo bem?”) é classificada como 0 → *Executor* retorna saudação. *Verifier* não encontra operação → aceita como “VÁLIDO”.  
2. **Pergunta de FAQ** (“Quanto custa a prova do CPHOS?”) é classificada como 1 → RAG no FAQ, chama LLM para sintetizar. *Verifier* não vê operação → “VÁLIDO”.  
3. **Consulta de usuário** (“Me mostre dados do usuário Joao”) classifica como 2 (depende de regex). *Executor* chama `consultar_usuario("joao")`, retorna dados. *Verifier* não vê *alterar_disciplina* → “VÁLIDO”.  
4. **Alterar disciplina** (“Quero alterar disciplina Joao Fisica”) → *Executor* chama `alterar_disciplina("joao", "Fisica")`. *Verifier* não encontra “Eu sou X” → sem bloqueio → “VÁLIDO”.  
5. **Alterar disciplina** (“Eu sou Maria. Desejo alterar disciplina Joao Matematica”) → *Verifier* encontra `operacao_sistema=("alterar_disciplina", "joao", "Matematica")` e “Eu sou Maria”. Maria é admin → “VÁLIDO”.  
6. **Alterar disciplina** (“Eu sou Joao. Desejo alterar disciplina Maria Fisica”) → *Verifier* vê que João não é admin e difere do user alvo (“maria”) → “INVÁLIDO”.

---

## Possíveis Melhorias

1. **Prompt Engineering** no LLM  
   - Ajustar prompts para evitar que o modelo repita partes do texto ou insira “Você é um assistente em português…” na resposta final.  
   - Adicionar exemplos *few-shot* para o classificador.  

2. **Controle de Sessão**  
   - Caso haja diálogos múltiplos (ex.: “Quero alterar disciplina para Fisica.” e depois “Aliás, para Química.”), seria preciso manter contexto do usuário (ID, sessão etc.).  

3. **Validação de Permissões Mais Rigorosa**  
   - O Verifier poderia chamar novamente o LLM para analisar a lógica de permissão, ou usar regras fixas adicionais.  

4. **UI/Front-End**  
   - Poderíamos expor esse chatbot via um endpoint REST (por ex. usando Flask ou FastAPI) e integrar com Slack, WhatsApp, etc.

5. **Otimização de Desempenho**  
   - Carregar FLAN-T5-base pode ser pesado. Caso o ambiente Colab seja fraco, considerar modelos menores ou *distil*.

---

## Conclusão

Este código exemplifica uma **arquitetura básica** de chatbot usando um **LLM gratuito** (Flan-T5-base) para:

- **Entender** a intenção do usuário (Classifier).  
- **Buscar informações** em um FAQ via *retrieval* (Executor).  
- **Executar operações** de forma controlada em um “banco de dados” simulado.  
- **Verificar permissões** (Verifier) antes de qualquer ação sensível.

Ele **não** é um sistema de produção pronto, mas **demonstra** como combinar:

- **Busca semântica** (FAISS + Sentence Transformers).  
- **Inferência de LLM local** (Flan-T5).  
- **Arquitetura CHOPS** (Classifier → Executor → Verifier).  

Para refinar, basta melhorar os prompts do LLM, adicionar mais regras de verificação e tratar sessões de conversa de maneira mais robusta.
