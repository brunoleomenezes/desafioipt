# -*- coding: utf-8 -*-
"""

Exemplo de Chatbot (Classifier → Executor → Verifier) usando:
- Busca semântica com sentence-transformers + FAISS
- LLM gratuito (FLAN-T5-base) da Hugging Face Transformers

Autor: Bruno Leonardo Santos Menezes

"""

# ==================================
# Passo 0: Instalações e imports
# ==================================
#!pip install sentence-transformers faiss-cpu transformers accelerate --quiet

import faiss
import numpy as np
import re

# Transformers (para LLM local)
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, pipeline

# ==================================
# Passo 1: Carregar FAQ e 'banco de dados' simulado
# ==================================

# Exemplo de "corpus" (guia/FAQ) em português
FAQ_TEXTS = [
    "Para adicionar estudantes, vá até o painel 'Estudantes' e clique em 'Novo'.",
    "A prova do CPHOS custa R$ 0, pois é oferecida gratuitamente.",
    "Para visualizar notas, acesse o menu 'Notas' e selecione o usuário desejado.",
    "Para alterar a disciplina de correção, utilize a configuração no painel de correções."
]

# Exemplo de "base de dados" (simulada) de usuários
USER_DB = {
    "maria": {"id": 1, "nome": "Maria", "disciplina": "Física", "adm": True},
    "joao":  {"id": 2, "nome": "João",  "disciplina": "Matemática", "adm": False},
}

# ==================================
# Passo 2: Indexação semântica do FAQ (Sentence Transformers + FAISS)
# ==================================
from sentence_transformers import SentenceTransformer

# Carrega modelo de embeddings
embedding_model = SentenceTransformer('sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2')

# Cria embeddings do FAQ
faq_embeddings = embedding_model.encode(FAQ_TEXTS)

dimension = faq_embeddings.shape[1]
index = faiss.IndexFlatL2(dimension)
index.add(faq_embeddings)

def retrieve_faq_responses(query, top_k=1):
    """Retorna o(s) trecho(s) de FAQ mais semelhante(s) à query."""
    query_emb = embedding_model.encode([query])
    distances, indices = index.search(query_emb, top_k)
    results = []
    for i in range(top_k):
        faq_idx = indices[0, i]
        results.append(FAQ_TEXTS[faq_idx])
    return results

# ==================================
# Passo 3: Criação de um "LLM local" com FLAN-T5-base (gratuito)
# ==================================
model_name = "google/flan-t5-base"
tokenizer = AutoTokenizer.from_pretrained(model_name)
llm_model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

# Pipeline de "text2text-generation"
llm_pipeline = pipeline("text2text-generation", model=llm_model, tokenizer=tokenizer)

def call_llm(prompt, max_new_tokens=64):
    """
    Função utilitária para 'dialogar' com o FLAN-T5-base.
    """

    resp = llm_pipeline(prompt, max_new_tokens=max_new_tokens, do_sample=False)
    # 'resp' é uma lista de dicionários [{'generated_text': '...'}]
    return resp[0]["generated_text"].strip()


# ==================================
# Passo 4: Definindo API simulada do sistema
# ==================================
def consultar_usuario(username):
    """Retorna dados do usuário ou None."""
    username = username.lower()
    return USER_DB.get(username, None)

def alterar_disciplina(username, nova_disciplina):
    """Altera a disciplina associada a certo usuário, se ele existir."""
    username = username.lower()
    if username in USER_DB:
        USER_DB[username]["disciplina"] = nova_disciplina
        return True
    return False

def is_admin(username):
    """Verifica se o usuário é admin."""
    user = USER_DB.get(username.lower())
    if user:
        return user["adm"]
    return False


# ==================================
# Passo 5: Classificador (C) usando LLM
# ==================================
def classify_user_text_llm(user_text):
    """
    Usa FLAN-T5-base para categorizar a pergunta do usuário em:
     0 - básico/trivial
     1 - FAQ/guia
     2 - ação no sistema
    """
    # Prompt simples para FLAN-T5:
    prompt = f"""
Classifique a solicitação abaixo em um número [0,1,2]:
- 0 se é algo trivial ou saudação
- 1 se é algo que requer conteúdo do FAQ
- 2 se requer uma ação/consulta no sistema (ex: alterar disciplina, etc.)

Pergunta: {user_text}

Responda apenas o número.
"""
    result = call_llm(prompt, max_new_tokens=10)
    # Tenta extrair um número (0,1,2)
    nums = re.findall(r"[012]", result)
    if nums:
        return int(nums[0])
    # fallback
    return 0


# ==================================
# Passo 6: Executor (E)
# - Se classificado como 0 => resposta trivial
# - Se classificado como 1 => busca no FAQ + chama LLM para gerar resposta final
# - Se classificado como 2 => tenta interpretar se é "consultar usuario" ou "alterar disciplina"
# ==================================
def executor(user_text, classification):
    operacao_sistema = None
    resposta = ""

    if classification == 0:
        # Retorno trivial
        resposta = "Olá, como posso te ajudar?"
        return resposta, operacao_sistema

    elif classification == 1:
        # RAG no FAQ + síntese com LLM
        trechos = retrieve_faq_responses(user_text, top_k=2)
        context_faq = "\n".join(trechos)

        prompt = f"""
Você é um assistente em português. Alguém perguntou: "{user_text}"
Aqui estão trechos do FAQ potencialmente relevantes:
{context_faq}

Com base nesses trechos, gere uma resposta curta e objetiva em português.
        """
        resposta = call_llm(prompt, max_new_tokens=60)
        return resposta, operacao_sistema

    else:
        # Classificação 2 => ação no sistema
        # Heurística rápida p/ identificar "consultar usuario" vs "alterar disciplina"
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
            user_alvo = match_disc.group(2).lower()
            nova = match_disc.group(3)
            ok = alterar_disciplina(user_alvo, nova)
            if ok:
                resposta = f"Disciplina de {user_alvo} alterada para {nova}."
            else:
                resposta = f"Não consegui alterar disciplina de {user_alvo}."
            operacao_sistema = ("alterar_disciplina", user_alvo, nova)

        if not resposta:
            resposta = "Parece ser algo relacionado ao sistema, mas não entendi a solicitação."

        return resposta, operacao_sistema


# ==================================
# Passo 7: Verificador (V) usando LLM para checar permissão
# ==================================
def verifier(user_text, resposta, operacao_sistema):
    """
    Verifica se a operação é válida.
    - Se "alterar_disciplina" e o solicitante não for admin ou não for o mesmo user => bloquear.
    """
    if not operacao_sistema:
        return True, None  # se não há operação, consideramos válido

    if operacao_sistema[0] == "alterar_disciplina":
        user_alvo = operacao_sistema[1]
        # Tenta extrair quem solicitou "Eu sou X"
        match_solicitante = re.search(r"eu sou\s+(\w+)", user_text.lower())
        if match_solicitante:
            solicitante = match_solicitante.group(1).lower()
            if not is_admin(solicitante) and solicitante != user_alvo:
                return False, "Você não tem privilégios para alterar a disciplina de outra pessoa."
    return True, None


# ==================================
# Passo 8: Pipeline final
# ==================================
def chatbot_pipeline(user_text):
    # 1. Classificador
    c = classify_user_text_llm(user_text)

    # 2. Executor
    resposta, operacao = executor(user_text, c)

    # 3. Verificador
    valido, motivo = verifier(user_text, resposta, operacao)
    if valido:
        return f"[Resposta VÁLIDA]\n{resposta}"
    else:
        return f"[Resposta INVÁLIDA]\nMotivo: {motivo}\nPor favor, reformule ou tente outra ação."


# ==================================
# Passo 9: Testes
# ==================================
print(chatbot_pipeline("Oi, tudo bem?"))
print(chatbot_pipeline("Quanto custa a prova do CPHOS?"))
print(chatbot_pipeline("Me mostre dados do usuário Joao"))
print(chatbot_pipeline("Quero alterar disciplina Joao Fisica"))
print(chatbot_pipeline("Eu sou Maria. Desejo alterar disciplina Joao Matematica"))
print(chatbot_pipeline("Eu sou Joao. Desejo alterar disciplina Maria Fisica"))
