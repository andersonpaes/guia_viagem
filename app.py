"""
Aplicativo Streamlit para o projeto **Guia de Viagem Inteligente**.

Este aplicativo reproduz a funcionalidade do notebook fornecido, mas expõe
os recursos por meio de uma interface de chat. O usuário faz perguntas
relacionadas à sua viagem e recebe respostas contextualizadas geradas por
uma pipeline de Geração Aumentada por Recuperação (RAG). A pipeline
combina recuperação de documentos a partir de um repositório vetorial
mantido em memória com um modelo de linguagem (LLM) executado pela Groq
para produzir respostas detalhadas e precisas em português do Brasil.

Para executar este aplicativo localmente instale as dependências
necessárias (consulte ``requirements.txt`` ou o notebook) e execute:

    streamlit run app.py

O aplicativo espera um arquivo PDF chamado ``guia_viagens.pdf`` em uma
pasta ``guia`` relativa ao diretório de trabalho atual. Esse PDF deve conter
informações turísticas para popular o índice de RAG. Você também deve
fornecer a chave de API ``GROQ_API_KEY`` para utilizar o LLM da Groq.
Utilize um arquivo ``.env`` na raiz do projeto para configurar essa
variável.
"""

import os
import re
import json
import time
from typing import List, Dict, Any, Optional

import streamlit as st
from dotenv import load_dotenv, find_dotenv
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.documents import Document
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
import numpy as np

try:
    from langchain_huggingface import HuggingFaceEmbeddings  # type: ignore
except ImportError:

    from langchain_community.embeddings import HuggingFaceEmbeddings  # type: ignore
    import warnings
    warnings.warn(
        "A classe HuggingFaceEmbeddings foi deprecada em LangChain 0.2.2. Instale o pacote 'langchain-huggingface' "
        "e importe-a como 'from langchain_huggingface import HuggingFaceEmbeddings' para evitar este aviso.",
        category=DeprecationWarning,
        stacklevel=2,
    )


def _format_context(docs: List[Document], max_chars: int = 4000) -> str:
    """Formata os documentos recuperados para inclusão no prompt.

    Cada documento é precedido por um índice e, quando disponível, a fonte
    (metadado ``source``). O tamanho total do contexto concatenado é
    limitado por ``max_chars`` para manter o prompt em um tamanho
    razoável. Quando não há metadados, apenas o índice do trecho é
    exibido.

    Parâmetros:
        docs: lista de documentos recuperados.
        max_chars: número máximo de caracteres permitidos no contexto.

    Retorna:
        Uma string formatada contendo os trechos selecionados dos
        documentos.
    """
    parts: List[str] = []
    total = 0
    for i, d in enumerate(docs, 1):
        text = d.page_content.strip()
        meta = d.metadata if isinstance(d.metadata, dict) else {}
        src = meta.get("source")
        header = f"[{i}] Fonte: {src}\n" if src else f"[{i}]\n"
        block = header + text + "\n"
        if total + len(block) > max_chars:
            break
        parts.append(block)
        total += len(block)
    return "\n".join(parts).strip()


def _has_sufficient_evidence(
    docs: List[Document],
    min_hits: int = 1,
    min_chars: int = 200,
    score_key: Optional[str] = "score",
    min_score: float = 0.3,
) -> bool:
    """Verifica se os documentos recuperados fornecem evidência suficiente.

    As heurísticas seguem as do notebook original: deve haver pelo menos
    ``min_hits`` documentos; a soma dos tamanhos de todos os documentos
    deve exceder ``min_chars``; e, quando houver pontuações de
    similaridade, pelo menos um trecho deve superar ``min_score``. Esses
    critérios ajudam a evitar alucinações quando a base de conhecimento
    carece de informações relevantes.

    Parâmetros:
        docs: lista de documentos recuperados.
        min_hits: quantidade mínima de documentos necessária.
        min_chars: comprimento total mínimo dos documentos.
        score_key: nome do metadado onde podem estar as pontuações de similaridade.
        min_score: pontuação mínima de similaridade aceitável.

    Retorna:
        ``True`` se houver evidência suficiente; caso contrário, ``False``.
    """
    if not docs or len(docs) < min_hits:
        return False
    # total characters across all retrieved documents
    total_chars = sum(len(getattr(d, "page_content", "") or "") for d in docs)
    # check similarity scores if present
    if score_key:
        scores: List[float] = []
        for d in docs:
            meta = getattr(d, "metadata", {}) or {}
            val = meta.get(score_key)
            if isinstance(val, (int, float)):
                scores.append(float(val))
            elif isinstance(val, list) and val and isinstance(val[0], (int, float)):
                scores.append(float(val[0]))
        if scores and max(scores) < min_score:
            return False
    return total_chars >= min_chars or len(docs) >= max(min_hits, 2)


def responder_rag(
    pergunta: str,
    retriever: Any,
    llm: Any,
    k: int = 4,
    min_hits: int = 1,
    min_chars: int = 200,
    score_key: Optional[str] = "score",
    min_score: float = 0.3,
    system_prefix: str = "",
    user_template: str = "",
) -> str:
    """Responde a uma pergunta usando uma pipeline RAG.

    Esta função recupera trechos de documentos relevantes usando o
    ``retriever`` fornecido e constrói um prompt para o modelo de linguagem
    (LLM). Caso não haja evidência suficiente, de acordo com as
    heurísticas em ``_has_sufficient_evidence``, retorna uma mensagem de
    desculpa padrão.

    Parâmetros:
        pergunta: pergunta do usuário.
        retriever: objeto com o método ``get_relevant_documents``.
        llm: modelo de linguagem utilizado para gerar a resposta.
        k: número de documentos principais a recuperar.
        min_hits: quantidade mínima de documentos exigida.
        min_chars: quantidade mínima de caracteres exigida.
        score_key: chave dos metadados para pontuações de similaridade.
        min_score: pontuação mínima aceitável de similaridade.
        system_prefix: instruções de sistema a serem colocadas antes do conteúdo do usuário.
        user_template: modelo usado para inserir a pergunta e o contexto.

    Retorna:
        A resposta gerada como string.
    """

    no_data_message = (
        "Desculpe, não encontrei essa informação na minha base de dados vetorizada (RAG). "
        "Sugiro realizar uma **nova atualização do RAG** com documentos que contenham esse conteúdo."
    )
    try:
        docs = retriever.get_relevant_documents(pergunta)[:k]
    except Exception as e:
        return f"{no_data_message} (Falha no retriever: {e})"
    if not _has_sufficient_evidence(docs, min_hits, min_chars, score_key, min_score):
        return no_data_message
    context = _format_context(docs)
    user_msg = user_template.format(question=pergunta, context=context)

    final_prompt = f"{system_prefix}\n\n{user_msg}"
    try:
        response = llm.invoke(final_prompt)

        if hasattr(response, "content"):
            return response.content
        if isinstance(response, dict) and "content" in response:
            return response["content"]
        return str(response)
    except Exception as e:
        return f"{no_data_message} (Falha no LLM: {e})"


def itinerary_chain(query: str, retriever: Any, llm: Any) -> str:
    """Cadeia especializada para geração de roteiros de viagem."""
    return responder_rag(
        pergunta=query,
        retriever=retriever,
        llm=llm,
        system_prefix=(
            "Você é um assistente que **somente** pode responder com base no CONTEXTO fornecido abaixo.\n"
            "Se a resposta não estiver no contexto, diga explicitamente que **não consta na base de dados** "
            "e **sugira uma atualização do RAG**. Não invente detalhes fora do contexto.\n\n"
            "Regras:\n"
            "1) Use apenas fatos presentes no CONTEXTO.\n"
            "2) Se faltar evidência suficiente, responda com a mensagem padrão de ausência.\n"
            "3) Seja conciso e cite trechos do contexto quando útil.\n\n"
            "Você é um assistente de viagens que responde em português do Brasil, com tom claro e útil. "
            "Gere um roteiro detalhado, organizado por dias e períodos. Considere perfil do viajante (cultural, "
            "gastronômico, aventura), duração, cidade e preferências."
        ),
        user_template=(
            "PERGUNTA:\n{question}\n\n"
            "CONTEXTO (trechos do RAG):\n{context}\n\n"
            "Responda **apenas** com base no CONTEXTO. Se não estiver no CONTEXTO, "
            "diga que não consta na base e sugira atualização do RAG. Inclua dicas práticas e alternativas em caso de chuva."
        ),
    )


def logistics_chain(query: str, retriever: Any, llm: Any) -> str:
    """Cadeia especializada para perguntas sobre transporte e logística."""
    return responder_rag(
        pergunta=query,
        retriever=retriever,
        llm=llm,
        system_prefix=(
            "Você é um assistente que **somente** pode responder com base no CONTEXTO fornecido abaixo.\n"
            "Se a resposta não estiver no contexto, diga explicitamente que **não consta na base de dados** "
            "e **sugira uma atualização do RAG**. Não invente detalhes fora do contexto.\n\n"
            "Regras:\n"
            "1) Use apenas fatos presentes no CONTEXTO.\n"
            "2) Se faltar evidência suficiente, responda com a mensagem padrão de ausência.\n"
            "3) Seja conciso e cite trechos do contexto quando útil.\n\n"
            "Você é um assistente de viagens que responde em português do Brasil, com tom claro e útil. "
            "Responda sobre transporte, acomodação, deslocamentos e custos aproximados quando possível."
        ),
        user_template=(
            "PERGUNTA:\n{question}\n\n"
            "CONTEXTO (trechos do RAG):\n{context}\n\n"
            "Responda **apenas** com base no CONTEXTO. Se não estiver no CONTEXTO, "
            "diga que não consta na base e sugira atualização do RAG."
        ),
    )


def localinfo_chain(query: str, retriever: Any, llm: Any) -> str:
    """Cadeia especializada para informações locais sobre atrações, restaurantes etc."""
    return responder_rag(
        pergunta=query,
        retriever=retriever,
        llm=llm,
        system_prefix=(
            "Você é um assistente que **somente** pode responder com base no CONTEXTO fornecido abaixo.\n"
            "Se a resposta não estiver no contexto, diga explicitamente que **não consta na base de dados** "
            "e **sugira uma atualização do RAG**. Não invente detalhes fora do contexto.\n\n"
            "Regras:\n"
            "1) Use apenas fatos presentes no CONTEXTO.\n"
            "2) Se faltar evidência suficiente, responda com a mensagem padrão de ausência.\n"
            "3) Seja conciso e cite trechos do contexto quando útil.\n\n"
            "Você é um assistente de viagens que responde em português do Brasil, com tom claro e útil. "
            "Forneça informações específicas sobre atrações, horários, restaurantes e eventos."
        ),
        user_template=(
            "PERGUNTA:\n{question}\n\n"
            "CONTEXTO (trechos do RAG):\n{context}\n\n"
            "Responda **apenas** com base no CONTEXTO. Se não estiver no CONTEXTO, "
            "diga que não consta na base e sugira atualização do RAG."
        ),
    )


def translation_chain(query: str, retriever: Any, llm: Any) -> str:
    """Cadeia especializada para traduções e frases úteis."""

    system_msg = (
        "Você é um guia de tradução para viagens que responde em português do Brasil. "
        "Forneça frases úteis traduzidas e transliterações quando cabível."
    )
    try:
        prompt = f"{system_msg}\n\nUsuário: {query}"
        response = llm.invoke(prompt)
        if hasattr(response, "content"):
            return response.content
        if isinstance(response, dict) and "content" in response:
            return response["content"]
        return str(response)
    except Exception as e:
        return (
            "Desculpe, não encontrei essa informação na minha base de dados vetorizada (RAG). "
            f"(Falha no LLM: {e})"
        )


def route_query(query: str, llm: Any) -> Dict[str, Any]:
    """Classifica a consulta do usuário em uma das rotas disponíveis.

    O roteador devolve um objeto JSON contendo a rota escolhida, uma
    justificativa breve e uma versão normalizada da consulta. Quando o
    parse falha, a rota padrão é ``"info-local"``.
    """
    router_prompt = ChatPromptTemplate.from_template(
        """
        Você é um roteador de intenções. Dado o texto do usuário, responda com um JSON válido com as chaves:
        - route: uma das opções ["roteiro-viagem", "logistica-transporte", "info-local", "traducao-idiomas"]
        - reasoning: breve justificativa (1–2 frases)
        - normalized_query: reescreva a consulta de forma clara e completa

        Exemplos:
        - "roteiro cultural em Paris por 3 dias" -> route="roteiro-viagem"
        - "como chegar ao Coliseu?" -> route="logistica-transporte"
        - "horário do Louvre e preço" -> route="info-local"
        - "frases básicas em japonês" -> route="traducao-idiomas"

        Usuário: {query}
        """
    )
    prompt = router_prompt.format(query=query)

    raw = (llm | StrOutputParser()).invoke(prompt)
    try:
        match = re.search(r"\{[\s\S]*\}", raw)
        data = json.loads(match.group(0) if match else raw)
    except Exception:
        data = {
            "route": "info-local",
            "reasoning": "fallback",
            "normalized_query": query,
        }
    return data


def routerchain_invoke(query: str, retriever: Any, llm: Any) -> Dict[str, Any]:
    """Executa o roteador e a cadeia apropriada para uma dada consulta do usuário."""
    decision = route_query(query, llm)
    route = decision.get("route", "info-local")
    norm_q = decision.get("normalized_query", query)

    route_to_fn = {
        "roteiro-viagem": itinerary_chain,
        "logistica-transporte": logistics_chain,
        "info-local": localinfo_chain,
        "traducao-idiomas": translation_chain,
    }
    chain_fn = route_to_fn.get(route, localinfo_chain)
    answer = chain_fn(norm_q, retriever, llm)
    return {
        "route": route,
        "normalized_query": norm_q,
        "answer": answer,
        "router_reasoning": decision.get("reasoning", ""),
    }


@st.cache_resource(show_spinner=False)
def load_resources() -> Dict[str, Any]:
    """Carrega o LLM, os documentos, os embeddings e configura o recuperador.

    Os recursos são armazenados em cache durante a sessão do Streamlit para que
    a inicialização pesada (leitura do PDF e criação de embeddings) ocorra
    apenas uma vez. Em vez de utilizar um repositório vetorial externo como
    o Pinecone, calculamos os embeddings localmente e usamos um recuperador
    em memória baseado na similaridade de cosseno.

    Retorna:
        Um dicionário contendo o LLM e o recuperador.
    """

    load_dotenv(find_dotenv())

    llm = ChatGroq(model_name="llama-3.3-70b-versatile", temperature=0)

    folder_path = "guia"
    filename = "guia_viagens.pdf"
    file_path = os.path.join(folder_path, filename)
    if not os.path.exists(file_path):
        raise FileNotFoundError(
            f"Arquivo PDF não encontrado em {file_path}. Coloque o arquivo guia_viagens.pdf na pasta 'guia'."
        )
    loader = PyPDFLoader(file_path)
    documents = loader.load()

    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    docs = splitter.split_documents(documents)

    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

    try:
        # Extrair apenas o conteúdo de cada documento para embedder
        texts = [d.page_content for d in docs]
        # Gerar embeddings para todos os documentos; retorna lista de listas de floats
        doc_embeddings = embeddings.embed_documents(texts)
        # Converter para array NumPy
        doc_vectors = np.array(doc_embeddings, dtype=float)
        # Normalizar cada vetor para comprimento 1 (importante para cosseno)
        doc_vectors_norm = doc_vectors / np.linalg.norm(doc_vectors, axis=1, keepdims=True)
    except Exception as e:
        raise RuntimeError(
            f"Erro ao gerar embeddings dos documentos: {e}"
        ) from e

    class SimpleRetriever:
        """Recuperador simples baseado em busca de similaridade de cosseno.

        Este recuperador armazena os embeddings dos documentos em memória e,
        dada uma pergunta, calcula o embedding da consulta usando o mesmo
        modelo. Em seguida, calcula o produto interno entre o vetor da
        consulta e todos os vetores normalizados dos documentos para
        determinar a proximidade (equivalente à similaridade de cosseno) e
        retorna os documentos com maiores pontuações.
        """

        def __init__(self, docs: List[Document], vectors: np.ndarray, emb_model: Any, k: int = 6) -> None:
            self.docs = docs
            self.vectors = vectors
            self.emb_model = emb_model
            self.k = k

        def get_relevant_documents(self, query: str) -> List[Document]:
            try:
                q_vec = self.emb_model.embed_query(query)
            except Exception as e:
                raise RuntimeError(f"Erro ao gerar embedding da consulta: {e}") from e
            # Converter para numpy array e normalizar
            q_vec = np.array(q_vec, dtype=float)
            q_vec_norm = q_vec / np.linalg.norm(q_vec)
            # Calcular similaridades (produto interno) com vetores normalizados dos documentos
            scores = np.dot(self.vectors, q_vec_norm)
            # Selecionar índices com maiores pontuações
            top_indices = np.argsort(scores)[::-1][: self.k]
            return [self.docs[i] for i in top_indices]

    # Criar instância do recuperador com k=6
    retriever = SimpleRetriever(docs, doc_vectors_norm, embeddings, k=6)
    return {"llm": llm, "retriever": retriever}


def main() -> None:
    """Ponto de entrada para o aplicativo Streamlit."""
    st.set_page_config(page_title="Guia de Viagem Inteligente", page_icon="🌍")
    st.title("Guia de Viagem Inteligente")

    try:
        resources = load_resources()
    except Exception as e:
        st.error(f"Erro ao carregar recursos: {e}")
        st.stop()
        return  
    llm = resources["llm"]
    retriever = resources["retriever"]

    if "messages" not in st.session_state:
        st.session_state.messages = []

    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    user_query = st.chat_input("Pergunte sobre viagens, roteiro, logística, informações locais ou traduções:")
    if user_query:

        st.session_state.messages.append({"role": "user", "content": user_query})
        with st.chat_message("user"):
            st.markdown(user_query)

        result = routerchain_invoke(user_query, retriever, llm)
        answer = result.get("answer", "Desculpe, algo deu errado ao gerar a resposta.")
        route = result.get("route", "info-local")
        reasoning = result.get("router_reasoning", "")

        reply = answer.strip()

        debug_info = f"\n\n*Rota escolhida:* `{route}`"
        if reasoning:
            debug_info += f"\n*Motivo:* {reasoning}"
        reply += debug_info
        # Append assistant's message to history
        st.session_state.messages.append({"role": "assistant", "content": reply})
        with st.chat_message("assistant"):
            st.markdown(reply)


if __name__ == "__main__":
    main()