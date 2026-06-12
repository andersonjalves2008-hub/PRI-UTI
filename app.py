import os
import streamlit as st
from dotenv import load_dotenv
from google import genai

load_dotenv()

client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))

st.set_page_config(
    page_title="PRI-UTI",
    page_icon="🏥",
    layout="wide"
)

st.title("🏥 PRI-UTI")
st.subheader("Sistema Inteligente de Priorização para Admissão em UTI")

st.markdown("---")
st.caption("PRI-UTI v1.0 • Desenvolvido por Anderson José Alves")

if "caso" not in st.session_state:
    st.session_state.caso = ""

def limpar():
    st.session_state.caso = ""
    st.session_state.resposta = ""

caso = st.text_area(
    "Cole a evolução clínica:",
    height=350,
    key="caso"
)

col1, col2 = st.columns([1, 1])

with col1:
    analisar = st.button("🔍 ANALISAR")

with col2:
    st.button("🧹 LIMPAR", on_click=limpar)

if analisar:
    if not caso.strip():
        st.warning("Cole um caso clínico antes de analisar.")
    else:
        try:
            with open("prompts/priorizacao.txt", "r", encoding="utf-8") as f:
                prompt_sistema = f.read()

            prompt_final = f"{prompt_sistema}\n\nCASO CLÍNICO:\n{caso}"

            resposta = client.models.generate_content(
                model="gemini-2.5-flash",
                contents=prompt_final
            )

            st.session_state.resposta = resposta.text

        except Exception as e:
            st.error(f"Erro ao analisar o caso: {e}")

if "resposta" in st.session_state and st.session_state.resposta:
    st.success(st.session_state.resposta)
