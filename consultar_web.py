# -*- coding: utf-8 -*-
import os
import json
import re
import colorama
from dotenv import load_dotenv
from langchain_google_genai import GoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_chroma import Chroma
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from datetime import datetime
import streamlit as st
import requests  # Para obtener la IP y geolocalización

# --- Configuración Inicial ---
colorama.init(autoreset=True)
load_dotenv()

# --- Carga de Modelos y Base de Datos (con caché de Streamlit) ---
@st.cache_resource
def load_resources():
    if "GOOGLE_API_KEY" not in os.environ:
        st.error("Error: La variable de entorno GOOGLE_API_KEY no está configurada.")
        st.stop()
    
    llm = GoogleGenerativeAI(model="models/gemini-2.5-pro")
    vectorstore = Chroma(
        persist_directory="./chroma_db",
        embedding_function=GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    )
    return llm, vectorstore

llm, vectorstore = load_resources()

# --- Lógica de GERARD (sin cambios) ---
prompt = ChatPromptTemplate.from_template("""
--- INICIO DE INSTRUCCIONES DE PERSONALIDAD ---
1. ROL Y PERSONA: Eres "GERARD", un analista de IA que encuentra patrones en textos.
2. CONTEXTO: Analizas archivos .srt sobre temas espirituales y narrativas ocultas.
--- REGLA DE FORMATO DE SALIDA (LA MÁS IMPORTANTE) ---
Tu única forma de responder es generando un objeto JSON. Tu respuesta DEBE ser un array de objetos JSON válido. Cada objeto debe tener dos claves: "type" y "content".
- "type" puede ser "normal" para texto regular, o "emphasis" para conceptos clave.
- "content" es el texto en sí.
EJEMPLO DE SALIDA OBLIGATORIA:
[
  {{ "type": "normal", "content": "El concepto principal es " }},
  {{ "type": "emphasis", "content": "la energía Crística" }},
  {{ "type": "normal", "content": ". (Fuente: archivo.srt, Timestamp: 00:01:23 --> 00:01:25)" }}
]
--- REGLA DE CITA ---
Incluye las citas de la fuente DENTRO del "content" de un objeto de tipo "normal". El formato es: `(Fuente: nombre_del_archivo.srt, Timestamp: HH:MM:SS --> HH:MM:SS)`.
Comienza tu labor, GERARD. Responde únicamente con el array JSON.
--- FIN DE INSTRUCCIONES DE PERSONALIDAD ---
Basándote ESTRICTAMENTE en las reglas y el contexto de abajo, responde la pregunta del usuario.
<contexto>
{context}
</contexto>
Pregunta del usuario: {input}
""")

def get_cleaning_pattern():
    texts_to_remove = [
        '[Spanish (auto-generated)]', '[DownSub.com]', '[Música]', '[Aplausos]'
    ]
    robust_patterns = [r'\[\s*' + re.escape(text[1:-1]) + r'\s*\]' for text in texts_to_remove]
    return re.compile(r'|'.join(robust_patterns), re.IGNORECASE)

cleaning_pattern = get_cleaning_pattern()

def format_docs_with_metadata(docs):
    formatted_strings = []
    for doc in docs:
        source_filename = os.path.basename(doc.metadata.get('source', 'Desconocido'))
        texts_to_remove_from_filename = ["[Spanish (auto-generated)]", "[DownSub.com]"]
        for text_to_remove in texts_to_remove_from_filename:
            source_filename = source_filename.replace(text_to_remove, "")
        source_filename = re.sub(r'\s+', ' ', source_filename).strip()
        cleaned_content = cleaning_pattern.sub('', doc.page_content)
        cleaned_content = re.sub(r'(\d{2}:\d{2}:\d{2}),\d{3}', r'\1', cleaned_content)
        cleaned_content = "\n".join(line for line in cleaned_content.split('\n') if line.strip())
        if cleaned_content:
            formatted_strings.append(f"Fuente: {source_filename}\nContenido:\n{cleaned_content}")
    return "\n\n---\n\n".join(formatted_strings)

retriever = vectorstore.as_retriever()
retrieval_chain = (
    {"context": retriever | format_docs_with_metadata, "input": RunnablePassthrough()}
    | prompt
    | llm
    | StrOutputParser()
)

# --- Funciones de Geolocalización y Registro ---
@st.cache_data
def get_user_location():
    try:
        response = requests.get('https://ipinfo.io/json', timeout=5)
        data = response.json()
        ip = data.get('ip', 'No disponible')
        city = data.get('city', 'Desconocida')
        country = data.get('country', 'Desconocido')
        return f"{city}, {country} (IP: {ip})"
    except Exception:
        return "Ubicación no disponible"

def get_clean_text_from_json(json_string):
    try:
        match = re.search(r'\[.*\]', json_string, re.DOTALL)
        if not match: return json_string
        data = json.loads(match.group(0))
        return "".join([item.get("content", "") for item in data])
    except:
        return json_string

def save_to_log(user, question, answer_json, location):
    clean_answer = get_clean_text_from_json(answer_json)
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    with open("gerard_log.txt", "a", encoding="utf-8") as f:
        f.write(f"--- Conversación del {timestamp} ---\n")
        f.write(f"Usuario: {user}\n")
        f.write(f"Ubicación: {location}\n")
        f.write(f"Pregunta: {question}\n")
        f.write(f"Respuesta de GERARD: {clean_answer}\n")
        f.write("="*40 + "\n\n")

# --- Interfaz de Usuario con Streamlit ---
st.set_page_config(page_title="GERARD", layout="centered")

# --- Avatares personalizados ---
user_avatar = "https://api.iconify.design/line-md/question-circle.svg?color=%2358ACFA"
assistant_avatar = "https://api.iconify.design/mdi/ufo-outline.svg?color=%238A2BE2"


# --- Estilos CSS y Título ---
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Orbitron:wght@700&display=swap');
.title-style {
    font-family: 'Orbitron', sans-serif;
    font-size: 5.5em;
    text-align: center;
    color: #8A2BE2; /* Violeta */
    padding-bottom: 20px;
}
.welcome-text {
    font-family: 'Orbitron', sans-serif;
    font-size: 2.5em;
    text-align: center;
    color: #28a745; /* Green */
    padding-bottom: 5px;
}
.sub-welcome-text {
    text-align: center;
    font-size: 1.1em;
    margin-top: -15px;
    padding-bottom: 20px;
}
.intro-text {
    text-transform: uppercase;
    text-align: center;
    color: #58ACFA; /* Azul claro */
    font-size: 2.6em;
    padding-bottom: 20px;
}
.loader-container {
    display: flex;
    align-items: center;
    justify-content: flex-start;
    padding-top: 5px;
}
.dot {
    height: 10px;
    width: 10px;
    margin: 0 3px;
    background-color: #8A2BE2; /* Violeta */
    border-radius: 50%;
    display: inline-block;
    animation: bounce 1.4s infinite ease-in-out both;
}
.dot:nth-child(1) { animation-delay: -0.32s; }
.dot:nth-child(2) { animation-delay: -0.16s; }
@keyframes bounce {
    0%, 80%, 100% { transform: scale(0); }
    40% { transform: scale(1.0); }
}

/* --- ¡NUEVA ANIMACIÓN CSS! --- */
.pulsing-q {
    font-size: 1.5em; /* 24px */
    color: red;
    font-weight: bold;
    animation: pulse 1.5s infinite;
}
@keyframes pulse {
    0% { transform: scale(1); opacity: 1; }
    50% { transform: scale(1.25); opacity: 0.75; }
    100% { transform: scale(1); opacity: 1; }
}
</style>
<div class="title-style">GERARD</div>
""", unsafe_allow_html=True)

location = get_user_location()

if 'user_name' not in st.session_state:
    st.session_state.user_name = ''
if 'messages' not in st.session_state:
    st.session_state.messages = []

if not st.session_state.user_name:
    st.markdown("""
    <p class="intro-text">
    Soy tu asistente especializado en analizar y encontrar el minuto y segundo exacto en los mensajes y meditaciones de los 9 maestros. Por favor, primero introduce tu nombre y te daré acceso para que hagas tus preguntas.
    </p>
    """, unsafe_allow_html=True)
    
    user_name_input = st.text_input("Tu Nombre", key="name_inputter", label_visibility="collapsed")
    if user_name_input:
        st.session_state.user_name = user_name_input.upper()
        st.rerun()
else:
    st.markdown(f"""
    <div class="welcome-text">BIENVENID@ {st.session_state.user_name}</div>
    <p class="sub-welcome-text">AHORA YA PUEDES REALIZAR TUS PREGUNTAS EN LA PARTE INFERIOR</p>
    """, unsafe_allow_html=True)

# --- Mostrar historial con avatares personalizados ---
for message in st.session_state.messages:
    avatar = user_avatar if message["role"] == "user" else assistant_avatar
    with st.chat_message(message["role"], avatar=avatar):
        st.markdown(message["content"], unsafe_allow_html=True)

# --- Input del usuario con avatares personalizados ---
if prompt_input := st.chat_input("Escribe tu pregunta aquí..."):
    if not st.session_state.user_name:
        st.warning("Por favor, introduce tu nombre para continuar.")
    else:
        # --- ¡AQUÍ ESTÁ EL CAMBIO! ---
        # Se reemplaza la imagen por un texto animado con CSS
        styled_prompt = f"""
        <div style="display: flex; align-items: center; justify-content: flex-start;">
            <span style="text-transform: uppercase; color: orange; margin-right: 8px; font-weight: bold;">{prompt_input}</span>
            <span class="pulsing-q">?</span>
        </div>
        """
        
        st.session_state.messages.append({"role": "user", "content": styled_prompt})
        
        with st.chat_message("user", avatar=user_avatar):
            st.markdown(styled_prompt, unsafe_allow_html=True)

        with st.chat_message("assistant", avatar=assistant_avatar):
            response_placeholder = st.empty()
            loader_html = """
            <div class="loader-container">
                <span class="dot"></span><span class="dot"></span><span class="dot"></span>
                <span style='margin-left: 10px; font-style: italic; color: #888;'>Pensando...</span>
            </div>
            """
            response_placeholder.markdown(loader_html, unsafe_allow_html=True)

            try:
                answer_json = retrieval_chain.invoke(prompt_input)
                save_to_log(st.session_state.user_name, prompt_input, answer_json, location)
                
                match = re.search(r'\[.*\]', answer_json, re.DOTALL)
                if not match:
                    st.error("La respuesta del modelo no fue un JSON válido.")
                    response_html = f'<p style="color:red;">{answer_json}</p>'
                else:
                    data = json.loads(match.group(0))
                    response_html = f'<strong style="color:#28a745;">{st.session_state.user_name}:</strong> '
                    for item in data:
                        content_type = item.get("type", "normal")
                        content = item.get("content", "")
                        if content_type == "emphasis":
                            response_html += f'<span style="color:yellow; background-color: #333; border-radius: 4px; padding: 2px 4px;">{content}</span>'
                        else:
                            content_html = re.sub(r'(\(.*?\))', r'<span style="color:#87CEFA;">\1</span>', content)
                            response_html += content_html
                
                response_placeholder.markdown(response_html, unsafe_allow_html=True)
                st.session_state.messages.append({"role": "assistant", "content": response_html})

            except Exception as e:
                response_placeholder.error(f"Ocurrió un error al procesar tu pregunta: {e}")

