import os
import re
import streamlit as st
from dotenv import load_dotenv
from google import genai
from google.genai import types


# =========================================================
# CONFIGURAÇÃO INICIAL
# =========================================================

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

st.markdown("""
<style>

.block-container{
    padding-top:0.8rem;
    padding-bottom:0rem;
}

h1{
    margin-bottom:0rem;
}

h2{
    margin-top:0rem;
    margin-bottom:0rem;
}

hr{
    margin-top:2px !important;
    margin-bottom:8px !important;
}

label{
    margin-bottom:2px !important;
}

</style>
""", unsafe_allow_html=True)

# =========================================================
# FUNÇÕES AUXILIARES
# =========================================================

def limpar():
    st.cache_data.clear()
    st.cache_resource.clear()

    for key in list(st.session_state.keys()):
        del st.session_state[key]

    st.rerun()


def carregar_prompt():
    caminho = "prompts/priorizacao.txt"

    if not os.path.exists(caminho):
        raise FileNotFoundError("Arquivo prompts/priorizacao.txt não encontrado.")

    with open(caminho, "r", encoding="utf-8") as f:
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


def extrair_texto_resposta(resposta):
    texto = ""

    try:
        if hasattr(resposta, "candidates") and resposta.candidates:
            for candidate in resposta.candidates:
                if hasattr(candidate, "content") and candidate.content:
                    if hasattr(candidate.content, "parts") and candidate.content.parts:
                        for part in candidate.content.parts:
                            if hasattr(part, "text") and part.text:
                                texto += part.text
    except Exception:
        texto = ""

    if not texto:
        texto = getattr(resposta, "text", "")

    return texto


def padronizar_resposta(texto):
    if not texto:
        return ""

    texto = texto.strip()
    texto = texto.replace("**", "")
    texto = texto.replace("```", "")
    texto = texto.replace("`", "")
    texto = texto.replace("...", "")
    texto = texto.replace("…", "")
    return texto.strip()


def validar_resposta(texto):
    if not texto:
        return False

    texto_upper = texto.upper()

    prioridades_validas = ["P1", "P2", "P3", "P4", "P5"]

    tem_prioridade = (
        "PRIORIDADE:" in texto_upper
        and any(p in texto_upper for p in prioridades_validas)
    )

    tem_justificativa = "JUSTIFICATIVA:" in texto_upper

    if not tem_prioridade or not tem_justificativa:
        return False

    justificativa = texto.split("JUSTIFICATIVA:", 1)[-1].strip()

    if len(justificativa) < 80:
        return False

    finais_invalidos = ("(", ",", ":", ";", "-", " e", " de", " com", " por", " em")

    if justificativa.endswith(finais_invalidos):
        return False

    if justificativa.endswith("...") or justificativa.endswith("…"):
        return False

    if justificativa.count("(") != justificativa.count(")"):
        return False

    if not justificativa.endswith("."):
        return False

    return True


def montar_prompt_final(prompt_sistema, caso):
    return f"""
{prompt_sistema}

CASO CLÍNICO:
{caso}

INSTRUÇÃO FINAL OBRIGATÓRIA:
Responda exatamente neste formato, sem texto adicional:

PRIORIDADE: P_

JUSTIFICATIVA:
Escreva uma justificativa completa, objetiva, em até 3 linhas, baseada apenas nos dados apresentados.

REGRAS FINAIS DA RESPOSTA:
- Não deixe frases incompletas.
- Não use reticências.
- Não deixe parênteses abertos.
- A justificativa deve terminar obrigatoriamente com ponto final.
- A resposta só será válida se contiver PRIORIDADE e JUSTIFICATIVA completas.
"""


def chamar_modelo(modelo, prompt_final):
    resposta = client.models.generate_content(
        model=modelo,
        contents=prompt_final,
        config=types.GenerateContentConfig(
            temperature=0,
            top_p=0.1,
            top_k=1,
            max_output_tokens=1500
        )
    )

    texto = extrair_texto_resposta(resposta)
    return padronizar_resposta(texto)


def analisar_caso(caso):
    prompt_sistema = carregar_prompt()
    prompt_final = montar_prompt_final(prompt_sistema, caso)

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
        for tentativa in range(2):
            try:
                texto = chamar_modelo(modelo, prompt_final)

                if validar_resposta(texto):
                    return texto, modelo

                ultimo_erro = Exception(
                    f"Modelo {modelo} retornou resposta incompleta ou fora do formato esperado:\n\n{texto}"
                )

            except Exception as e:
                ultimo_erro = e

                if erro_de_cota(e):
                    segundos = extrair_retry_delay(e)

                    if segundos:
                        erros_cota.append(
                            f"{modelo}: cota atingida. Tentar novamente em {segundos}s."
                        )
                    else:
                        erros_cota.append(
                            f"{modelo}: cota atingida."
                        )

                    break

                raise e

    if erros_cota:
        raise Exception("\n".join(erros_cota))

    raise ultimo_erro if ultimo_erro else Exception("Nenhum modelo retornou resposta válida.")


# =========================================================
# ESTADO DA SESSÃO
# =========================================================

if "caso" not in st.session_state:
    st.session_state.caso = ""

if "resposta" not in st.session_state:
    st.session_state.resposta = ""

if "modelo_usado" not in st.session_state:
    st.session_state.modelo_usado = ""


# =========================================================
# INTERFACE
# =========================================================

st.title("🏥 PRI-UTI")
st.subheader("Sistema Inteligente de Priorização para Admissão em UTI")

st.markdown(
    "<p style='font-size:13px; color:gray; margin-top:-12px; margin-bottom:4px;'>"
    "PRI-UTI v1.0 • Desenvolvido por Anderson José Alves - Qualimed"
    "</p>",
    unsafe_allow_html=True,
)

st.markdown("<hr style='margin-top:2px; margin-bottom:8px;'>", unsafe_allow_html=True)

col_caso, col_resultado = st.columns([1.15, 1])

with col_caso:

    st.markdown("### Digite ou cole a evolução clínica")

    caso = st.text_area(
        label="",
        height=300,
        key="caso",
        label_visibility="collapsed"
    )

    botao1, botao2 = st.columns(2)

    with botao1:
        analisar = st.button(
            "🔍 ANALISAR",
            use_container_width=True
        )

    with botao2:
        st.button(
            "🧹 LIMPAR",
            use_container_width=True,
            on_click=limpar
        )

with col_resultado:

    st.markdown("### Resultado")

    if st.session_state.resposta:

        resposta_html = (
            st.session_state.resposta
            .replace(
                "PRIORIDADE:",
                "<div style='font-size:32px;font-weight:700;margin-bottom:18px;'>PRIORIDADE:"
            )
            .replace(
    "JUSTIFICATIVA:",
    "</div><div style='font-size:22px;font-weight:700;margin-top:12px;margin-bottom:1px;'>JUSTIFICATIVA:</div>"
)
        )

        st.markdown(
            f"""
<div style="
border:1px solid #3b82f6;
border-radius:10px;
padding:12px 18px;
min-height:420px;
background-color:rgba(59,130,246,0.05);
overflow:hidden;
">

<div style="
margin:0;
padding:0;
display:flex;
flex-direction:column;
justify-content:flex-start;
align-items:flex-start;
height:100%;
font-size:19px;
line-height:1.8;
white-space:pre-wrap;
">

{resposta_html}

</div>

</div>
""",
            unsafe_allow_html=True,
        )

        st.caption(
            f"Análise concluída • Modelo utilizado: {st.session_state.modelo_usado}"
        )

    else:

        st.markdown(
            """
<div style="
border:1px solid #444;
border-radius:10px;
padding:12px 18px;
min-height:420px;
background-color:rgba(120,120,120,0.05);
display:flex;
align-items:flex-start;
justify-content:flex-start;
font-size:19px;
color:gray;
">

O resultado da análise aparecerá aqui.

</div>
""",
            unsafe_allow_html=True,
        )

# =========================================================
# ANÁLISE
# =========================================================

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
            st.rerun()

        except Exception as e:
            erro = str(e)

            if erro_de_cota(e):
                with col_resultado:
                    st.warning(
                        "⏳ Limite temporário ou diário da API Gemini atingido nos modelos disponíveis. "
                        "Aguarde a renovação da cota ou habilite faturamento no Google AI Studio."
                    )

                    with st.expander("Mostrar erro técnico"):
                        st.code(erro)

            elif "priorizacao.txt" in erro:
                with col_resultado:
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
                with col_resultado:
                    st.error(
                        "❌ Problema na chave da API Gemini. "
                        "Verifique se GEMINI_API_KEY está correta nos Secrets do Streamlit."
                    )

                    with st.expander("Mostrar erro técnico"):
                        st.code(erro)

            else:
                with col_resultado:
                    st.error("❌ Ocorreu um erro inesperado.")

                    with st.expander("Mostrar erro técnico"):
                        st.code(erro)
