import os
import streamlit as st
from dotenv import load_dotenv
from google import genai

# ==========================================
# CARREGA VARIÁVEIS
# ==========================================

load_dotenv()

def get_api_key():
    try:
        return st.secrets["GEMINI_API_KEY"]
    except Exception:
        return os.getenv("GEMINI_API_KEY")

api_key = get_api_key()

if not api_key:
    st.error("❌ GEMINI_API_KEY não encontrada.")
    st.stop()

client = genai.Client(api_key=api_key)

# ==========================================
# CONFIGURAÇÃO DA PÁGINA
# ==========================================

st.set_page_config(
    page_title="PRI-UTI",
    page_icon="🏥",
    layout="wide"
)

# ==========================================
# FUNÇÕES
# ==========================================

def limpar():

    for key in list(st.session_state.keys()):
        del st.session_state[key]

    st.rerun()


def carregar_prompt():

    with open(
        "prompts/priorizacao.txt",
        "r",
        encoding="utf-8"
    ) as arquivo:

        return arquivo.read()


def analisar(caso):

    prompt = carregar_prompt()

    prompt_final = (
        prompt
        + "\n\nCASO CLÍNICO:\n"
        + caso
    )

    resposta = client.models.generate_content(
        model="gemini-2.5-flash",
        contents=prompt_final
    )

    return resposta.text


# ==========================================
# ESTADO DA SESSÃO
# ==========================================

if "resposta" not in st.session_state:
    st.session_state.resposta = ""

if "caso" not in st.session_state:
    st.session_state.caso = ""

# ==========================================
# CABEÇALHO
# ==========================================

st.title("🏥 PRI-UTI")

st.subheader(
    "Sistema Inteligente de Priorização para Admissão em UTI"
)

st.caption(
    "by Anderson José Alves"
)

st.divider()

# ==========================================
# ÁREA DE TEXTO
# ==========================================

caso = st.text_area(
    "Cole a evolução clínica:",
    key="caso",
    height=350
)

# ==========================================
# BOTÕES
# ==========================================

col1, col2 = st.columns(2)

with col1:

    analisar_botao = st.button(
        "🔍 ANALISAR",
        use_container_width=True
    )

with col2:

    st.button(
        "🧹 LIMPAR",
        use_container_width=True,
        on_click=limpar
    )

# ==========================================
# ANÁLISE
# ==========================================

if analisar_botao:

    st.session_state.resposta = ""

    if caso.strip() == "":

        st.warning(
            "Cole um caso clínico antes de analisar."
        )

    else:

        try:

            with st.spinner("Analisando caso..."):

                resposta = analisar(caso)

                st.session_state.resposta = resposta

        except Exception as e:

            erro = str(e)

            st.session_state.resposta = ""

            if (
                "RESOURCE_EXHAUSTED" in erro
                or "429" in erro
                or "Quota exceeded" in erro
                or "rate limit" in erro.lower()
            ):

                st.warning(
                    "⏳ Limite temporário da API Gemini atingido.\n\n"
                    "Aguarde cerca de 1 minuto e tente novamente."
                )

                with st.expander(
                    "Mostrar erro técnico"
                ):
                    st.code(erro)

            elif (
                "GEMINI_API_KEY" in erro
                or "API key" in erro
                or "permission" in erro.lower()
            ):

                st.error(
                    "❌ Problema na chave da API Gemini."
                )

                with st.expander(
                    "Mostrar erro técnico"
                ):
                    st.code(erro)

            elif (
                "priorizacao.txt" in erro
            ):

                st.error(
                    "❌ Arquivo prompts/priorizacao.txt não encontrado."
                )

            else:

                st.error(
                    "❌ Ocorreu um erro inesperado."
                )

                with st.expander(
                    "Mostrar erro técnico"
                ):
                    st.code(erro)

# ==========================================
# RESULTADO
# ==========================================

if st.session_state.resposta != "":

    st.success(
        st.session_state.resposta
    )
