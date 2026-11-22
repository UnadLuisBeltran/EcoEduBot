import streamlit as st

from groq_backend import (
    generar_respuesta_groq,
    contextualizar_pregunta
)
from descargar_modelo import descargar_modelo

# Download the ONNX model before initializing the RAG system.
descargar_modelo()
from rag_backend import RecuperadorIPCC, construir_contexto


st.set_page_config(
    page_title="EcoEduBot",
    page_icon="🌍",
    layout="centered"
)


@st.cache_resource(show_spinner="Loading IPCC knowledge base...")
def cargar_recuperador():
    """Initialize the IPCC hybrid retriever."""
    return RecuperadorIPCC()


st.title("🌍 EcoEduBot")
st.subheader("Environmental Education Assistant")

st.write(
    "Ask questions about climate change, sustainability, "
    "and environmental education using evidence from "
    "the IPCC 2023 Synthesis Report."
)

try:
    recuperador = cargar_recuperador()
except Exception as error:
    st.error("The knowledge base could not be loaded.")
    st.exception(error)
    st.stop()

if "messages" not in st.session_state:
    st.session_state.messages = []

for mensaje in st.session_state.messages:
    with st.chat_message(mensaje["role"]):
        st.markdown(mensaje["content"])

pregunta = st.chat_input(
    "Ask an environmental question in English..."
)

if pregunta:
    historial = st.session_state.messages.copy()

    st.session_state.messages.append({
        "role": "user",
        "content": pregunta
    })

    with st.chat_message("user"):
        st.markdown(pregunta)

   with st.chat_message("assistant"):
        with st.spinner("Searching IPCC evidence..."):
            try:
                # Resolve conversational references before retrieval.
                consulta_rag = contextualizar_pregunta(
                    pregunta,
                    historial
                )

                # Retrieve IPCC evidence.
                fragmentos = recuperador.recuperar(
                    consulta_rag,
                    candidatos=20,
                    top_k=3
                )

                contexto = construir_contexto(fragmentos)

                respuesta = generar_respuesta_groq(
                    pregunta,
                    contexto,
                    historial
                )

                paginas = sorted({
                    f["pagina_pdf"]
                    for f in fragmentos
                })

                paginas_texto = ", ".join(
                    str(p) for p in paginas
                )

                respuesta_final = (
                    f"{respuesta}\n\n"
                    f"**Retrieved IPCC PDF pages:** "
                    f"{paginas_texto}"
                )

                st.markdown(respuesta_final)

                st.session_state.messages.append({
                    "role": "assistant",
                    "content": respuesta_final
                })

            except Exception as error:
                st.error(
                    "An error occurred while generating the answer."
                )
                st.exception(error)
