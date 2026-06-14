import os
import streamlit as st
from dotenv import load_dotenv
from google import genai

# =========================
# CONFIGURAÇÃO INICIAL
# =========================

load_dotenv()

def get_api_key():
    try:
        return st.secrets["GEMINI_API_KEY"]
    except Exception:
        return os.getenv("GEMINI_API_KEY")

api_key = get_api_key()

if not api_key:
    st.error("Chave GEMINI_API_KEY não encontrada. Configure no .env local ou nos Secrets do Streamlit.")
    st.stop()

client = genai.Client(api_key=api_key)

st.set_page_config(
    page_title="PRI-UTI",
    page_icon="🏥",
    layout="wide"
)

# =========================
# FUNÇÕES
# =========================

def limpar():
    st.session_state.clear()
    st.rerun()

def carregar_prompt():
    with open("prompts/priorizacao.txt", "r", encoding="utf-8") as f:
        return f.read()

def analisar_caso(caso):
    prompt_sistema = carregar_prompt()
    prompt_final = f"{prompt_sistema}\n\nCASO CLÍNICO:\n{caso}"

    resposta = client.models.generate_content(
        model="gemini-2.5-flash",
        contents=prompt_final
    )

    return resposta.text

# =========================
# INTERFACE
# =========================

st.title("🏥 PRI-UTI")
st.subheader("Sistema Inteligente de Priorização para Admissão em UTI")
st.caption("PRI-UTI v1.0 • Desenvolvido por Anderson José Alves")
st.markdown("---")

if "caso" not in st.session_state:
    st.session_state.caso = ""

if "resposta" not in st.session_state:
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

# =========================
# ANÁLISE
# =========================

if analisar:
    if not caso.strip():
        st.warning("Cole um caso clínico antes de analisar.")
    else:
        try:
            with st.spinner("Analisando o caso..."):
                st.session_state.resposta = analisar_caso(caso)

        except Exception as e:
            erro = str(e)

            if "429" in erro or "RESOURCE_EXHAUSTED" in erro:
                st.warning(
                    "⏳ Limite temporário da API Gemini atingido. "
                    "Aguarde aproximadamente 1 minuto e clique em LIMPAR para reiniciar a aplicação."
                )
            elif "prompts/priorizacao.txt" in erro:
                st.error(
                    "Arquivo prompts/priorizacao.txt não encontrado. "
                    "Verifique se a pasta prompts e o arquivo priorizacao.txt estão no GitHub."
                )
            else:
                st.error(f"Erro ao analisar o caso: {erro}")

# =========================
# RESULTADO
# =========================

if st.session_state.resposta:
    st.success(st.session_state.resposta)
