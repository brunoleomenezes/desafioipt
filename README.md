## Visão Geral do Fluxo

1. **Classifier** (Classificador)  
   Recebe o texto do usuário e determina se a questão é sobre:
   - **(0) Pergunta básica** (ex.: saudações, algo trivial que não necessita consultas).  
   - **(1) Pergunta sobre o guia/FAQ** (texto extenso e pré-cadastrado).  
   - **(2) Pergunta que requer acesso ao “sistema”** (simulação de chamadas de API para consultar ou alterar dados).  

2. **Executor**  
   Baseado na classificação, executa:  
   - Resposta “simples” se for trivial (categoria 0).  
   - *Retrieval* (busca) de trechos do guia/FAQ se for categoria 1.  
   - Chamadas de “API” (funções Python simuladas) se for categoria 2.  

3. **Verifier**  
   Verifica a resposta ou operação gerada pelo Executor. Se considerar inválida, retorna o motivo, e uma nova tentativa pode ser feita. Caso seja válida, devolve a resposta final ao usuário.

Esse fluxo reflete o design de muitos sistemas de atendimento que precisam, ao mesmo tempo, **responder perguntas** e **executar operações seguras** no back-end.

---

## Passo a Passo do Código

### Passo 0: Instalações e Imports (Google Colab)

```bash
!pip install sentence-transformers --quiet
!pip install faiss-cpu --quiet
```

- **sentence-transformers**: biblioteca para gerar *embeddings* de texto (e.g. *bert-like*).  
- **faiss-cpu**: biblioteca da Meta (Facebook AI) para fazer busca vetorial (similaridades).

Na sessão Python, importamos:

```python
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
import re
```

---

### Passo 1: Definição do “Corpus” (FAQ) e da “Base de Dados” Simulada

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

- **FAQ_TEXTS** simula um conjunto de trechos de um guia ou FAQ.
- **USER_DB** simula uma base de dados de usuários (mini-“sistema”), contendo alguns campos como `disciplina`, `adm` (se é administrador ou não) etc.

---

### Passo 2: Indexação do Corpus de FAQ (para busca semântica)

1. **Carregar modelo de embeddings**  
   Neste caso, usamos `paraphrase-multilingual-MiniLM-L12-v2` (multilíngue e leve).  

2. **Gerar embeddings do FAQ**  
   Convertendo cada texto do FAQ em um vetor numérico.

3. **Criar índice FAISS**  
   - Adicionamos as representações vetoriais de cada trecho de FAQ a um índice FAISS, que permite buscas por similaridade.

```python
model = SentenceTransformer('sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2')

faq_embeddings = model.encode(FAQ_TEXTS)
dimension = faq_embeddings.shape[1]
index = faiss.IndexFlatL2(dimension)
index.add(faq_embeddings)

def retrieve_faq_responses(query, top_k=1):
    query_emb = model.encode([query])
    distances, indices = index.search(query_emb, top_k)
    results = []
    for i in range(top_k):
        faq_idx = indices[0, i]
        results.append(FAQ_TEXTS[faq_idx])
    return results
```

- **retrieve_faq_responses(query, top_k)** retorna o(s) trecho(s) do FAQ mais próximos semântica ou contextualmente da pergunta.

---

### Passo 3: Funções Simuladas de “API do Sistema”

```python
def consultar_usuario(username):
    username = username.lower()
    return USER_DB.get(username, None)

def alterar_disciplina(username, nova_disciplina):
    username = username.lower()
    if username in USER_DB:
        USER_DB[username]["disciplina"] = nova_disciplina
        return True
    return False

def is_admin(username):
    user = USER_DB.get(username.lower())
    if user:
        return user["adm"]
    return False
```

Essas funções exemplificam chamadas comuns em sistemas:
- `consultar_usuario`: busca os dados no dicionário `USER_DB`.  
- `alterar_disciplina`: modifica um atributo do usuário.  
- `is_admin`: verifica se certo usuário é administrador.

---

### Passo 4: Classificador (Classifier)

```python
def classify_user_text(user_text):
    text_lower = user_text.lower()
    
    # Categoria 0: perguntas básicas
    if any(x in text_lower for x in ["oi", "olá", "bom dia", "boa tarde"]):
        return 0
    
    # Categoria 1: perguntas de FAQ (heurística)
    if re.search(r"(como)|(quanto)|(para que)|(custa)|(test)|(guia)", text_lower):
        return 1
    
    # Categoria 2: perguntas que envolvem API (disciplina, usuário, etc.)
    if re.search(r"(disciplina)|(mudar)|(usuário)|(user)|(alterar)|(admin)|(estudante)", text_lower):
        return 2
    
    # Padrão: consideramos categoria 0
    return 0
```

- **Objetivo**: Retornar um número (0, 1 ou 2) indicando se a pergunta do usuário se resolve com resposta básica, consulta ao FAQ ou chamadas de API.  
- É um exemplo didático com regex/palavras-chave. Em produção, pode-se usar LLMs ou modelos de classificação treinados.

---

### Passo 5: Executor (Executor)

```python
def executor(user_text, classification):
    operacao_sistema = None  
    resposta = ""
    
    if classification == 0:
        # Resposta básica
        resposta = "Olá! Como posso ajudar?"
    
    elif classification == 1:
        # Buscar no FAQ
        trechos = retrieve_faq_responses(user_text, top_k=1)
        resposta = f"Baseado no guia, encontrei: {trechos[0]}"
    
    elif classification == 2:
        # Decidir se é "consultar usuário" ou "alterar disciplina" (heurística regex)
        text_lower = user_text.lower()
        
        match_user = re.search(r"usuário\s+(\w+)", text_lower)
        if match_user:
            username = match_user.group(1)
            user_data = consultar_usuario(username)
            if user_data:
                resposta = f"Dados de {username}: {user_data}"
            else:
                resposta = f"Usuário {username} não encontrado."
        
        match_disc = re.search(r"(mudar|alterar)\s+disciplina\s+(\w+)\s+(\w+)", text_lower)
        if match_disc:
            old_or_user = match_disc.group(2)
            new_disc = match_disc.group(3)
            ok = alterar_disciplina(old_or_user, new_disc)
            if ok:
                resposta = f"Disciplina de {old_or_user} alterada para {new_disc}."
            else:
                resposta = f"Não consegui alterar disciplina de {old_or_user}."
            operacao_sistema = ("alterar_disciplina", old_or_user, new_disc)
        
        # Se não casou nenhum regex e não geramos resposta
        if resposta == "":
            resposta = "Parece ser algo relacionado ao sistema, mas não entendi a solicitação."
    
    return resposta, operacao_sistema
```

- **Fluxo**:  
  1. Se for classificação 0 → retorna algo simples.  
  2. Se for 1 → chama `retrieve_faq_responses` para buscar no FAQ.  
  3. Se for 2 → faz lógica para identificar e executar operações via API:
     - Consultar dados de usuário.  
     - Alterar disciplina.  

- Armazena `operacao_sistema` se houve alguma chamada efetiva, para que o Verifier possa analisar.

---

### Passo 6: Verificador (Verifier)

```python
def verifier(resposta, operacao_sistema, user_text):
    if not operacao_sistema:
        return True, None
    
    if operacao_sistema[0] == "alterar_disciplina":
        user_alvo = operacao_sistema[1]
        
        # Exemplo: extrair quem está fazendo o pedido
        match_solicitante = re.search(r"eu sou\s+(\w+)", user_text.lower())
        if match_solicitante:
            solicitante = match_solicitante.group(1)
            if not is_admin(solicitante) and solicitante != user_alvo.lower():
                return False, "Usuário não tem privilégios para alterar disciplina de outra pessoa."
    
    return True, None
```

- Se não houve operação (ou seja, `operacao_sistema` é `None`), consideramos “válido” por padrão.  
- Se for “alterar_disciplina”, checa se:
  - O solicitante é **admin** ou está alterando a própria disciplina.  
  - Caso contrário, retorna `(False, <motivo>)`, isto é, operação inválida.  

Em sistemas reais, o Verifier poderia ser muito mais complexo, verificando permissões, logs, limites de uso etc.

---

### Passo 7: Função “chatbot_pipeline” (União de Tudo)

```python
def chatbot_pipeline(user_text):
    # 1) Classifier
    c = classify_user_text(user_text)
    
    # 2) Executor
    resposta, operacao = executor(user_text, c)
    
    # 3) Verifier
    valido, motivo = verifier(resposta, operacao, user_text)
    
    if valido:
        return f"[Resposta VÁLIDA]\n{resposta}"
    else:
        return f"[Resposta INVÁLIDA]\nMotivo: {motivo}\nPor favor, reformule ou tente outra ação."
```

- Chamamos cada etapa em sequência:
  1. Classificador determina a categoria.  
  2. Executor gera resposta ou opera o sistema.  
  3. Verificador valida a saída final.  
- Se for válida, retorna resposta ao usuário; se inválida, retorna o motivo.

---

## Testes Práticos

Chamamos a função `chatbot_pipeline()` com exemplos de entrada:

```python
print(chatbot_pipeline("Oi, tudo bem?"))
print(chatbot_pipeline("Quanto custa a prova do CPHOS?"))
print(chatbot_pipeline("Me mostre dados do usuário Joao"))
print(chatbot_pipeline("Quero alterar disciplina Joao Fisica"))
print(chatbot_pipeline("Eu sou Maria. Gostaria de alterar disciplina Joao Matematica"))
print(chatbot_pipeline("Eu sou Joao. Desejo alterar disciplina Maria Fisica"))
```

**Possível saída**:  
```
[Resposta VÁLIDA]
Olá! Como posso ajudar?
[Resposta VÁLIDA]
Baseado no guia, encontrei: A prova do CPHOS custa R$ 0, pois é oferecida gratuitamente.
[Resposta VÁLIDA]
Dados de joao: {'id': 2, 'nome': 'João', 'disciplina': 'Matemática', 'adm': False}
[Resposta VÁLIDA]
Disciplina de joao alterada para fisica.
[Resposta VÁLIDA]
Disciplina de joao alterada para matematica.
[Resposta INVÁLIDA]
Motivo: Usuário não tem privilégios para alterar disciplina de outra pessoa.
Por favor, reformule ou tente outra ação.
```

Isso confirma que o fluxo funciona:
1. Perguntas triviais → Resposta básica.  
2. Perguntas sobre FAQ → Busca no corpus de FAQ.  
3. Operações no sistema → Chamada de API + verificação de permissões.  

---

## Pontos de Atenção e Possíveis Melhorias

1. **Heurísticas de Classificação**  
   - Atualmente, o *Classifier* usa expressões regulares e palavras-chave. Pode ser substituído por um modelo de classificação de textos usando LLM ou *fine-tuning*.  

2. **Regex para parsing de comandos**  
   - No *Executor*, identificar a intenção de “consultar usuário” ou “alterar disciplina” é feito via regex. Em produção, poderia haver um parser mais robusto ou uso de *prompt engineering*.  

3. **Persistência de Dados**  
   - Aqui, `USER_DB` é um simples dicionário. Em um sistema real, haveria persistência (banco relacional, NoSQL, etc.) e encapsulamento de transações.  

4. **Segurança e Permissões**  
   - O Verifier exemplifica como se poderia checar privilégios. Em produção, esse check poderia ser expandido para logs, limites de operação, auditoria etc.  

5. **LLMs Reais e Prompt Tuning**  
   - Para integrar GPT-4 ou GPT-3.5, seria preciso usar *prompts* customizados e organizar respostas em formato de JSON (para não cometer erros ao chamar APIs).  

6. **Manter Contexto (Conversational)**  
   - Caso haja múltiplas trocas de mensagem, poderia-se gerenciar sessões, estados de conversas e “lembrar” perguntas anteriores do usuário.  

---

## Conclusão

Este código **demonstra** a arquitetura **Classifier → Executor → Verifier** em um ambiente simplificado. Ele ilustra como combinar:

- **Busca vetorial (FAISS)** para responder perguntas com base em textos de FAQ,  
- **Chamadas de API** (funções Python que representam o sistema de back-end),  
- **Camada de verificação** para evitar ações incorretas ou sem permissão.  

O modelo resultante serve como **prova de conceito** de um chatbot híbrido que responde tanto perguntas baseadas em documentos quanto realiza operações seguras no sistema, aproximando-se de cenários reais de atendimento ao cliente.
