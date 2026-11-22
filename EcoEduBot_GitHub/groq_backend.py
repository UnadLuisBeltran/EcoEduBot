import streamlit as st
from groq import Groq


def generar_respuesta_groq(pregunta, contexto, historial=None):
    """Generate an answer using IPCC evidence."""

    cliente = Groq(api_key=st.secrets["GROQ_API_KEY"])

    mensajes = [
        {
            "role": "system",
            "content": (
                "You are EcoEduBot, an environmental education assistant. "
                "Answer in English using the retrieved IPCC evidence. "
                "Do not invent facts or references. "
                "If the evidence is insufficient, acknowledge the limitation."
            )
        }
    ]

    if historial:
        for mensaje in historial[-6:]:
            if (
                mensaje.get("role") in ("user", "assistant")
                and isinstance(mensaje.get("content"), str)
            ):
                mensajes.append({
                    "role": mensaje["role"],
                    "content": mensaje["content"]
                })

    mensajes.append({
        "role": "user",
        "content": (
            f"Retrieved IPCC evidence:\n{contexto}\n\n"
            f"Question: {pregunta}"
        )
    })

    respuesta = cliente.chat.completions.create(
        model="qwen/qwen3.8-27b",
        messages=mensajes,
        temperature=0.2,
        max_tokens=500
    )

    return respuesta.choices[0].message.content
    
def contextualizar_pregunta(pregunta, historial=None):
    """Rewrite follow-up questions for document retrieval."""

    if not historial:
        return pregunta

    cliente = Groq(
        api_key=st.secrets["GROQ_API_KEY"]
    )

    mensajes = [
        {
            "role": "system",
            "content": (
                "Rewrite the latest user question as a standalone "
                "English search query for an IPCC document retriever. "
                "Use the conversation history to resolve references "
                "such as 'the third impact', 'that issue', or 'it'. "
                "Preserve the original meaning. "
                "Do not answer the question. "
                "Do not invent information. "
                "Return only the rewritten question."
            )
        }
    ]

    for mensaje in historial[-6:]:
        if (
            mensaje.get("role") in ("user", "assistant")
            and isinstance(mensaje.get("content"), str)
        ):
            mensajes.append({
                "role": mensaje["role"],
                "content": mensaje["content"]
            })

    mensajes.append({
        "role": "user",
        "content": pregunta
    })

    respuesta = cliente.chat.completions.create(
        model="qwen/qwen3.8-27b",
        messages=mensajes,
        temperature=0,
        max_tokens=150
    )

    consulta = (
        respuesta.choices[0].message.content or ""
    ).strip()

    return consulta if consulta else pregunta

