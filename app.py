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
    st.error("❌ GEMINI_API_KEY não encontrada. Configure no .env ou nos Secrets do Streamlit.")
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
    for key in list(st.session_state.keys()):
        del st.session_state[key]
    st.rerun()

def carregar_prompt():
    with open("prompts/priorizacao.txt", "r", encoding="utf-8") as f:
        return f.read()

def erro_de_cota(erro):
    texto = str(erro)
    return (
        "RESOURCE_EXHAUSTED" in texto
        or "429" in texto
        or "Quota exceeded" in texto
        or "rate limit" in texto.lower()
    )

def analisar_caso(caso):
    prompt_sistema = carregar_prompt()
    prompt_final = f"{prompt_sistema}\n\nCASO CLÍNICO:\n{caso}"

    modelos = [
        "gemini-2.5-flash",
        "gemini-2.5-flash-lite"
    ]

    ultimo_erro = None

    for modelo in modelos:
        try:
            resposta = client.models.generate_content(
                model=modelo,
                contents=prompt_final
            )

            return resposta.text, modelo

        except Exception as e:
            ultimo_erro = e

            if erro_de_cota(e):
                continue
            else:
                raise e

    raise ultimo_erro

# =========================
# ESTADO DA SESSÃO
# =========================

if "caso" not in st.session_state:
    st.session_state.caso = ""

if "resposta" not in st.session_state:
    st.session_state.resposta = ""

if "modelo_usado" not in st.session_state:
    st.session_state.modelo_usado = ""

# =========================
# INTERFACE
# =========================

st.title("🏥 PRI-UTI")
st.subheader("Sistema Inteligente de Priorização para Admissão em UTI")
st.caption("PRI-UTI v1.0 • Desenvolvido por Anderson José Alves")
st.divider()

caso = st.text_area(
    "Cole a evolução clínica:",
    height=350,
    key="caso"
)

col1, col2 = st.columns(2)

with col1:
    analisar = st.button("🔍 ANALISAR", use_container_width=True)

with col2:
    st.button("🧹 LIMPAR", use_container_width=True, on_click=limpar)

# =========================
# ANÁLISE
# =========================

if analisar:
    st.session_state.resposta = ""
    st.session_state.modelo_usado = ""

    if not caso.strip():
        st.warning("Cole um caso clínico antes de analisar.")
    else:
        try:
            with st.spinner("Analisando caso..."):
                resposta, modelo_usado = analisar_caso(caso)

            st.session_state.resposta = resposta
            st.session_state.modelo_usado = modelo_usado

        except Exception as e:
            erro = str(e)

            if erro_de_cota(e):
                st.warning(
                    "⏳ Limite temporário ou diário da API Gemini atingido nos modelos disponíveis. "
                    "Aguarde a renovação da cota ou habilite faturamento no Google AI Studio."
                )

                with st.expander("Mostrar erro técnico"):
                    st.code(erro)

            elif "priorizacao.txt" in erro:
                st.error(
                    "❌ Arquivo prompts/priorizacao.txt não encontrado. "
                    "Verifique se a pasta prompts está no GitHub."
                )

            elif "API key" in erro or "GEMINI_API_KEY" in erro or "permission" in erro.lower():
                st.error(
                    "❌ Problema na chave da API Gemini. "
                    "Verifique se GEMINI_API_KEY está correta nos Secrets do Streamlit."
                )

                with st.expander("Mostrar erro técnico"):
                    st.code(erro)

            else:
                st.error("❌ Ocorreu um erro inesperado.")

                with st.expander("Mostrar erro técnico"):
                    st.code(erro)

# =========================
# RESULTADO
# =========================

if st.session_state.resposta:
    st.success(st.session_state.resposta)

    if st.session_state.modelo_usado:
        st.caption(f"Modelo utilizado: {st.session_state.modelo_usado}")
