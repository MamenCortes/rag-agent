# Chatbot Agéntico con RAG, Búsqueda Web y Guardarraíles
Practica final del taller de agentes NextDigital - CEU San Pablo

## Arquitectura del sistema

```
┌─────────────────────────────────────────────────────────────────┐
│                        INTERFAZ (Streamlit)                     │
│                  Chat UI con historial de mensajes              │
└──────────────────────────────┬──────────────────────────────────┘
                               │ mensaje del usuario
                               ▼
┌─────────────────────────────────────────────────────────────────┐
│                       GUARDARRAÍLES                             │
│  ¿La pregunta es relevante para la temática del documento?      │
│                                                                 │
│       SÍ ──────────────────────────────────────────────┐        │
│       NO → respuesta de rechazo educada                │        │
└────────────────────────────────────────────────────────┼────────┘
                                                         │
                                                         ▼
┌─────────────────────────────────────────────────────────────────┐
│                        AGENTE ReAct                             │
│              (LangChain + ChatOpenAI, temperatura 0)            │
│                                                                 │
│  Razonamiento:                                                  │
│  1. ¿Tengo suficiente contexto? → usar buscar_en_documento      │
│  2. ¿Necesito info actual/no encontrada? → usar buscar_en_web   │
│  3. Formular respuesta final                                    │
└──────────┬──────────────────────────────┬───────────────────────┘
           │                              │
           ▼                              ▼
┌──────────────────────┐    ┌─────────────────────────────────────┐
│   TOOL: RAG          │    │   TOOL: Búsqueda Web                │
│                      │    │                                     │
│  Base de datos       │    │   DuckDuckGo / Tavily               │
│  vectorial (Chroma)  │    │   para información actual           │
│                      │    │   no presente en el documento       │
│  ← documentos PDF    │    │                                     │
└──────────────────────┘    └─────────────────────────────────────┘
           │                              │
           └──────────────┬───────────────┘
                          ▼
              respuesta final con fuentes
                          │
                          ▼
┌─────────────────────────────────────────────────────────────────┐
│                     INTERFAZ (Streamlit)                        │
│            Respuesta mostrada en el chat + fuentes              │
└─────────────────────────────────────────────────────────────────┘
```

---

## Enunciado

### Descripción general

Desarrolla un **chatbot agéntico con interfaz web** (Streamlit) especializado en la temática de los documentos que hayas indexado. El chatbot debe ser capaz de responder preguntas consultando primero su base de datos vectorial y, si no encuentra información suficiente, acudir a internet. Para mantener la coherencia del asistente, debe incorporar guardarraíles que rechacen preguntas ajenas al dominio del documento.

---

### Requisitos funcionales

**1. Interfaz (Streamlit)**
- Pantalla de chat con historial de mensajes persistente durante la sesión.
- Campo de entrada de texto para el usuario.
- Las respuestas del agente deben mostrarse de forma clara, indicando si la información proviene del documento o de internet.
- Opcional: muestra el razonamiento interno del agente (cadena de pensamiento) en un expander o panel lateral.

**2. Guardarraíles (topic guard)**
- Antes de invocar al agente, evalúa si la pregunta es relevante para la temática del documento.
- Si la pregunta está fuera del dominio (por ejemplo, el documento es sobre inteligencia artificial y el usuario pregunta sobre recetas de cocina), el chatbot debe responder con un mensaje educado explicando en qué temáticas puede ayudar, **sin llamar al agente ni consumir herramientas**.
- Implementa el guardarraíl como una llamada al LLM con un system prompt clasificador que devuelva `"relevante"` o `"irrelevante"`.

**3. Herramienta RAG (`buscar_en_documento`)**
- Carga uno o varios PDFs y los indexa en una base de datos vectorial (Chroma u otra).
- Expón el retriever como una `@tool` de LangChain siguiendo el patrón visto en clase.
- El agente debe usarla como primera opción ante cualquier pregunta relevante.

**4. Herramienta de búsqueda web (`buscar_en_internet`)**
- Usa DuckDuckGoSearchRun (o Tavily si dispones de API key) como herramienta de respaldo.
- El agente solo debe llamarla cuando la búsqueda en el documento no haya proporcionado una respuesta satisfactoria, o cuando la pregunta requiera información actualizada que no puede estar en el documento.

**5. Agente**
- Crea un agente ReAct con `create_react_agent` que tenga acceso a ambas herramientas.
- El system prompt del agente debe reforzar su especialización: definir quién es, en qué es experto y el orden de preferencia de las herramientas.

---

### Requisitos técnicos

| Componente | Tecnología |
|---|---|
| Interfaz | Streamlit |
| Agente | LangChain `create_react_agent` |
| LLM | OpenAI `gpt-5.4-nano` o superior |
| Embeddings | `text-embedding-3-small` |
| Base vectorial | Chroma (persistente o en memoria) |
| Búsqueda web | DuckDuckGoSearchRun |
| Variables de entorno | `python-dotenv` |

---

### Pistas y referencias

- Revisa `sesion_2/5_loop_agente.py` para ver cómo se combinan RAG y búsqueda web en un agente ReAct.
- Revisa `sesion_2/3_memoria_rag.py` para entender el patrón `history_aware_retriever` que necesitarás para mantener el historial en el chatbot.
- Para arrancar la app: `uv run streamlit run practica_final/app.py`
- El historial de mensajes de Streamlit se gestiona con `st.session_state.messages`.
- Para que el guardarraíl sea robusto, usa `temperature=0` y pide al modelo que responda **únicamente** con la palabra `relevante` o `irrelevante`.
- Si el agente tarda en responder, considera usar `st.spinner()` mientras procesa.
