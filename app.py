import os
import re
import streamlit as st
from dotenv import load_dotenv
from google import genai
from google.genai import types

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


def extrair_retry_delay(erro):
    texto = str(erro)

    match = re.search(r"retryDelay['\"]?: ['\"]?(\d+)s", texto)
    if match:
        return int(match.group(1))

    match = re.search(r"Please retry in ([\d\.]+)s", texto)
    if match:
        return int(float(match.group(1)))

    return None


def validar_resposta(texto):
    if not texto:
        return False

    texto_upper = texto.upper()

    if "PRIORIDADE:" not in texto_upper:
        return False

    if "JUSTIFICATIVA:" not in texto_upper:
        return False

    prioridades_validas = ["P1", "P2", "P3", "P4", "P5"]

    return any(p in texto_upper for p in prioridades_validas)


def padronizar_resposta(texto):
    texto = texto.strip()

    # Remove markdown excessivo
    texto = texto.replace("**", "")
    texto = texto.replace("```", "")

    return texto.strip()


def analisar_caso(caso):
    prompt_sistema = carregar_prompt()

    prompt_final = f"""
{prompt_sistema}

CASO CLÍNICO:
{caso}

LEMBRE-SE:
Responda exclusivamente no formato:

PRIORIDADE: P_

JUSTIFICATIVA:
Máximo de 3 linhas.
"""

    modelos = [
        "gemini-2.5-flash",
        "gemini-2.5-flash-lite",
        "gemini-2.5-pro",
        "gemini-2.0-flash",
        "gemini-2.0-flash-lite"
    ]

    erros_cota = []
    ultimo_erro = None

    for modelo in modelos:
        try:
            resposta = client.models.generate_content(
                model=modelo,
                contents=prompt_final,
                config=types.GenerateContentConfig(
                    temperature=0,
                    top_p=0.1,
                    top_k=1,
                    max_output_tokens=300
                )
            )

            texto = getattr(resposta, "text", "")

            if not texto or not texto.strip():
                ultimo_erro = Exception(f"Modelo {modelo} não retornou texto.")
                continue

            texto = padronizar_resposta(texto)

            if validar_resposta(texto):
                return texto, modelo

            ultimo_erro = Exception(
                f"Modelo {modelo} retornou resposta fora do formato esperado:\n\n{texto}"
            )
            continue

        except Exception as e:
            ultimo_erro = e

            if erro_de_cota(e):
                segundos = extrair_retry_delay(e)
                if segundos:
                    erros_cota.append(f"{modelo}: cota atingida. Tentar novamente em {segundos}s.")
                else:
                    erros_cota.append(f"{modelo}: cota atingida.")
                continue

            # Se não for erro de cota, não troca de modelo: mostra o erro real.
            raise e

    if erros_cota:
        raise Exception("\n".join(erros_cota))

    raise ultimo_erro if ultimo_erro else Exception("Nenhum modelo retornou resposta válida.")


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
st.caption("PRI-UTI v2.0 • Desenvolvido por Anderson José Alves - Qualimed")
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

            elif (
                "api key" in erro.lower()
                or "gemini_api_key" in erro.lower()
                or "permission" in erro.lower()
                or "unauthorized" in erro.lower()
            ):
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
