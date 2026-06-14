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
    texto = str(erro).lower()
    return (
        "resource_exhausted" in texto
        or "429" in texto
        or "quota exceeded" in texto
        or "rate limit" in texto
        or "exceeded your current quota" in texto
    )


def listar_modelos_disponiveis():
    modelos = []

    try:
        for model in client.models.list():
            nome = getattr(model, "name", "")

            if not nome:
                continue

            nome = nome.replace("models/", "")

            # Evita modelos que geralmente não servem para texto clínico
            ignorar = [
                "embedding",
                "imagen",
                "veo",
                "aqa",
                "tts",
                "native-audio",
                "live",
                "preview-image"
            ]

            if any(x in nome.lower() for x in ignorar):
                continue

            modelos.append(nome)

    except Exception:
        modelos = []

    prioridade = [
        "gemini-2.5-pro",
        "gemini-2.5-flash",
        "gemini-2.5-flash-lite",
        "gemini-2.0-flash",
        "gemini-2.0-flash-lite",
        "gemini-1.5-pro",
        "gemini-1.5-flash",
        "gemini-1.5-flash-8b"
    ]

    modelos_ordenados = []

    for m in prioridade:
        if m in modelos:
            modelos_ordenados.append(m)

    for m in modelos:
        if m not in modelos_ordenados:
            modelos_ordenados.append(m)

    return modelos_ordenados


def analisar_caso(caso):
    prompt_sistema = carregar_prompt()
    prompt_final = f"{prompt_sistema}\n\nCASO CLÍNICO:\n{caso}"

    modelos = listar_modelos_disponiveis()

    if not modelos:
        modelos = [
            "gemini-2.5-pro",
            "gemini-2.5-flash",
            "gemini-2.5-flash-lite",
            "gemini-2.0-flash",
            "gemini-2.0-flash-lite",
            "gemini-1.5-pro",
            "gemini-1.5-flash",
            "gemini-1.5-flash-8b"
        ]

    ultimo_erro = None

    for modelo in modelos:
        try:
            resposta = client.models.generate_content(
                model=modelo,
                contents=prompt_final
            )

            texto = getattr(resposta, "text", "")

            if texto and texto.strip():
                return texto.strip()

        except Exception as e:
            ultimo_erro = e
            continue

    raise ultimo_erro if ultimo_erro else Exception("Nenhum modelo disponível retornou resposta válida.")


# =========================
# ESTADO DA SESSÃO
# =========================

if "caso" not in st.session_state:
    st.session_state.caso = ""

if "resposta" not in st.session_state:
    st.session_state.resposta = ""


# =========================
# INTERFACE
# =========================

st.title("🏥 PRI-UTI")
st.subheader("Sistema Inteligente de Priorização para Admissão em UTI")
st.caption("PRI-UTI v1.0 • Desenvolvido por Anderson José Alves - Qualimed")
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

    if not caso.strip():
        st.warning("Cole um caso clínico antes de analisar.")
    else:
        try:
            with st.spinner("Analisando caso..."):
                resposta = analisar_caso(caso)

            st.session_state.resposta = resposta

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

            elif "api key" in erro.lower() or "gemini_api_key" in erro.lower() or "permission" in erro.lower():
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
    st.caption("Análise concluída.")
