import os
from dotenv import load_dotenv

import streamlit as st

from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_chroma import Chroma
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.tools import DuckDuckGoSearchRun
from langchain_core.tools import tool
from langgraph.prebuilt import create_react_agent


load_dotenv()

RUTA_PDF = os.path.join(
    os.path.dirname(__file__),
    "data",
    "documento.pdf",
)


@st.cache_resource
def crear_retriever():
    #1. Cargar el PDF
    loader = PyPDFLoader(RUTA_PDF)
    documentos = loader.load()
    print(f"Páginas cargadas: {len(documentos)}")

    #2. Trocear (chunking)
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50,
        separators=["\n\n", "\n", ". ", " ", ""],
    )

    chunks = splitter.split_documents(documentos)
    print(f"Chunks generados: {len(chunks)}")

    #3. Crear vector store
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")

    vector_store = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        collection_name="rag_agent_documentos",
    )

    return vector_store.as_retriever(search_kwargs={"k": 3})


def crear_tool_documento():
    retriever = crear_retriever()

    @tool
    #convertimos el buscador documental en una tool de LangChain para que el agente pueda usarla
    def buscar_en_documento(consulta: str) -> str:
        """Busca información en los documentos indexados. Úsala siempre primero."""
        docs = retriever.invoke(consulta)

        if not docs:
            return "No se encontró información relevante en el documento."

        resultado = []
        for i, doc in enumerate(docs, 1):
            fuente = doc.metadata.get("source", "documento")
            pagina = doc.metadata.get("page", "?")
            resultado.append(
                f"[Fuente documento {i}: {fuente}, página {pagina}]\n{doc.page_content}"
            )

        return "\n\n".join(resultado)

    return buscar_en_documento


def crear_tool_web():
    buscador = DuckDuckGoSearchRun()

    @tool
    def buscar_en_internet(consulta: str) -> str:
        """Busca información actual en internet cuando el documento no sea suficiente."""
        return buscador.invoke(consulta)

    return buscar_en_internet


def clasificar_relevancia(pregunta: str) -> str:
    #creamos un LLM separado para clasificar la pregunta antes de llamar al agente.
    llm_guard = ChatOpenAI(model="gpt-5.4-nano", temperature=0)

    prompt = f"""
Eres un clasificador de preguntas.

El documento trata sobre inteligencia artificial, machine learning, deep learning,
LLMs, aplicaciones de IA, ética, privacidad e impacto tecnológico.

Clasifica la pregunta del usuario como:
- relevante: si está relacionada con la temática anterior.
- irrelevante: si no está relacionada.

Responde únicamente con una palabra: relevante o irrelevante.

Pregunta:
{pregunta}
"""

    respuesta = llm_guard.invoke(prompt).content.strip().lower()

    if "irrelevante" in respuesta:
        return "irrelevante"

    return "relevante"


@st.cache_resource
def crear_agente():
    llm = ChatOpenAI(model="gpt-5.4-nano", temperature=0)

    buscar_doc = crear_tool_documento()
    buscar_web = crear_tool_web()

    system_prompt = """
Eres un chatbot agéntico especializado en inteligencia artificial.

Reglas:
1. Usa siempre primero buscar_en_documento.
2. Si el documento no contiene información suficiente, usa buscar_en_internet.
3. Si usas información del documento, indícalo claramente.
4. Si usas internet, indícalo claramente.
5. Responde de forma clara, didáctica y breve.
6. No inventes fuentes.
"""

    agent = create_react_agent(
        llm,
        tools=[buscar_doc, buscar_web],
        prompt=system_prompt,
    )

    return agent


#Interfaz con Streamlit
# 1. Inicializar Streamlit y configurar la página
st.set_page_config(page_title="Chatbot Agéntico RAG", page_icon="🤖")
st.title("🤖 Chatbot Agéntico con RAG, Web y Guardarraíles")

# 2. Inicializar el historial de mensajes en el estado de sesión
# usar st.session_state.messages para mantener el historial durante la sesión
if "messages" not in st.session_state:
    st.session_state.messages = []

#Cada vez que Streamlit recarga la página, vuelve a pintar los mensajes antiguos.
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

#Entrada del usuario: st.chat_input para enviar preguntas al agente
pregunta = st.chat_input("Pregunta sobre el documento...")

if pregunta:
    #Guardar la pregunta del usuario en el historial de mensajes
    st.session_state.messages.append({"role": "user", "content": pregunta})

    with st.chat_message("user"):
        st.markdown(pregunta)

    with st.chat_message("assistant"):
        with st.spinner("Analizando la pregunta..."):
            #1. Guardarraíles: clasificar la pregunta antes de llamar al agente
            relevancia = clasificar_relevancia(pregunta)

            #2. Llamar al agente solo si la pregunta es relevante. Si no, responder con un mensaje de error.
            if relevancia == "irrelevante":
                respuesta = (
                    "Lo siento, solo puedo responder preguntas relacionadas con "
                    "la temática del documento: inteligencia artificial, machine learning, "
                    "deep learning, LLMs, aplicaciones de IA y aspectos éticos."
                )
                st.markdown(respuesta)

            else:
                agente = crear_agente()

                historial = [
                    {"role": m["role"], "content": m["content"]}
                    for m in st.session_state.messages
                ]

                resultado = agente.invoke({"messages": historial})
                respuesta = resultado["messages"][-1].content

                st.markdown(respuesta)

    st.session_state.messages.append({"role": "assistant", "content": respuesta})