from langchain_core.messages import HumanMessage, SystemMessage
from utils.agent_state import AgentState
from utils.llm import chat_llm
from utils.debug_time import time_check


@time_check
def incompleteAccountAgent(state: AgentState):
    info = "\n--- Incomplete Account ---"
    print(info)

    emailAccountUser = state["emailAccountUser"]
    loginAccountStatus = state["loginAccountStatus"]

    prompt = f"""
        Anda adalah seorang pengirim pesan informasi Undiksha.
        Tugas Anda untuk memberitahu pengguna bahwa:
        Mohon maaf, saya tidak dapat membantu menangani akun SSO atau Google Undiksha Anda.
        Berikut informasi dari yang anda diberikan:
        - Email Account User: {emailAccountUser} (jika null = ganti menjadi "Tidak disebutkan")
        - Login Account Status: {loginAccountStatus} (jika true = ganti menjadi "Sudah login", jika false = ganti menjadi "Belum login")
        Petunjuk untuk Pengguna:
        - Email valid dari Undiksha "@undiksha.ac.id" atau "@student.undiksha.ac.id"
        - Pastikan akun google sudah login di email/gmail/google/hp/laptop/komputer.
        - Beritahu kesalahan pengguna.
        Minta pengguna untuk mengisi format pengajuan jika ingin reset password dan kirim disini:
        - Email: Masukkan Email (contoh: shavira@undiksha.ac.id atau shavira@student.undiksha.ac.id)
        - Login Status: Masukkan Status Login (Contoh: Sudah Login / Belum Login di Perangkat)
    """
    messages = [
        SystemMessage(content=prompt),
        HumanMessage(content=state["accountQuestion"])
    ]
    response = chat_llm(messages)

    agentOpinion = {
        "question": state["accountQuestion"],
        "answer": response
    }
    state["finishedAgents"].add("incompleteAccount_agent") 
    
    return {"answerAgents": [agentOpinion]}
