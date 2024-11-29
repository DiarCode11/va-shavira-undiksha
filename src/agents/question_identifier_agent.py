import re
from utils.debug_time import time_check
from utils.expansion import query_expansion, CONTEXT_ABBREVIATIONS
from utils.agent_state import AgentState
from langchain_core.messages import HumanMessage, SystemMessage
from utils.llm import chat_llm


@time_check
def questionIdentifierAgent(state: AgentState):
    info = "\n--- QUESTION IDENTIFIER ---"
    print(info)

    original_question = state['question']
    cleaned_question = re.sub(r'\n+', ' ', original_question)
    expanded_question = query_expansion(cleaned_question, CONTEXT_ABBREVIATIONS)
    state["question"] = expanded_question

    prompt = """
        Anda adalah seoarang pemecah pertanyaan pengguna.
        Tugas Anda sangat penting. Klasifikasikan atau parsing pertanyaan dari pengguna untuk dimasukkan ke variabel sesuai konteks.
        Tergantung pada jawaban Anda, akan mengarahkan ke agent yang tepat.
        Ada 5 konteks diajukan:
        - GENERAL_AGENT - Berkaitan dengan segala informasi umum mahasiswa, dosen, pegawai, civitas akademika universitas dll dan jika ada yang bertanya tentang dirimu atau sapaan.
        - NEWS_AGENT - Hanya jika pertanyaan mengandung kata "berita" atau "news".
        - ACCOUNT_AGENT - Bekaitan dengan reset ulang lupa password hanya pada akun email Universitas Pendidikan Ganesha (Undiksha) atau ketika user lupa dengan password email undiksha di gmail (google) atau user lupa password login di SSO Undiksha, jika hanya masalah cara merubah password itu masuk ke general.
        - KELULUSAN_AGENT - Pertanyaan terkait pengecekan status kelulusan bagi pendaftaran calon mahasiswa baru yang telah mendaftar di Undiksha, biasanya pertanyaan pengguna berisi nomor pendaftaran dan tanggal lahir.
        - KTM_AGENT - Hanya jika pertanyaan mengandung kata "ktm" atau "nim". Jika menyebutkan "nip" maka itu general.
        Kemungkinan pertanyaannya berisi lebih dari 1 variabel konteks yang berbeda, buat yang ada sesuai dengan konteks saja (jika tidak ada jangan dibuat).
        Jawab pertanyaan dan sertakan pertanyaan pengguna yang sesuai dengan kategori dengan contoh seperti ({"GENERAL_AGENT": "pertanyaan relevan terkait general", "NEWS_AGENT": "hanya jika pertanyaan mengandung kata "berita" atau "news"", "ACCOUNT_AGENT": "pertanyaan relevan terkait lupa password akun", "KELULUSAN_AGENT": "pertanyaan relevan terkait kelulusan", "KTM_AGENT": "hanya jika pertanyaan mengandung kata "ktm" atau "nim"."}).
        Buat dengan format data JSON tanpa membuat key baru.
    """
    messagesTypeQuestion = [
        SystemMessage(content=prompt),
        HumanMessage(content=expanded_question),
    ]
    responseTypeQuestion = chat_llm(messagesTypeQuestion).strip().lower()

    state["question_type"] = responseTypeQuestion
    print("\nPertanyaan:", expanded_question)

    total_agents = 0
    if "general_agent" in state["question_type"]:
        total_agents += 3
    if "news_agent" in state["question_type"]:
        total_agents += 1
    if "account_agent" in state["question_type"]:
        total_agents += 2
    if "kelulusan_agent" in state["question_type"]:
        total_agents += 2
    if "ktm_agent" in state["question_type"]:
        total_agents += 2
    state["totalAgents"] = total_agents
    print(f"DEBUG: Total agents bertugas: {state['totalAgents']}")
    pattern = r'"(.*?)":\s*"(.*?)"'
    matches = re.findall(pattern, responseTypeQuestion)
    result_dict = {key: value for key, value in matches}

    state["generalQuestion"] = result_dict.get("general_agent", None)
    state["newsQuestion"] = result_dict.get("news_agent", None)
    state["accountQuestion"] = result_dict.get("account_agent", None)
    state["kelulusanQuestion"] = result_dict.get("kelulusan_agent", None)
    state["ktmQuestion"] = result_dict.get("ktm_agent", None)
    
    print(f"DEBUG: generalQuestion: {state['generalQuestion']}")
    print(f"DEBUG: newsQuestion: {state['newsQuestion']}")
    print(f"DEBUG: accountQuestion: {state['accountQuestion']}")
    print(f"DEBUG: kelulusanQuestion: {state['kelulusanQuestion']}")
    print(f"DEBUG: ktmQuestion: {state['ktmQuestion']}")

    return state