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
