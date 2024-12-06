from langchain_core.messages import HumanMessage, SystemMessage
from utils.agent_state import AgentState
from utils.llm import chat_llm
from utils.api_undiksha import show_reset_sso
from utils.debug_time import time_check


@time_check
def resetAccountAgent(state: AgentState):
    info = "\n--- Reset Account ---"
    print(info)

    state["emailAccountUser"]
    state["loginAccountStatus"]
    reset_sso_info = show_reset_sso(state)

    try:
        email = reset_sso_info["email"]
        tipe_user = reset_sso_info["tipe_user"]
        is_email_sent = reset_sso_info["is_email_sent"]
        prompt = f"""
            Anda adalah seorang pengirim pesan informasi Undiksha.
            Tugas Anda untuk memberitahu pengguna bahwa:
            Selamat, pengajuan proses reset password akun SSO Undiksha berhasil!
            Berikut informasi akun Pengguna:
            - Email Account User: {email} (jika null = ganti menjadi "Tidak disebutkan")
            - Tipe User: {tipe_user} (jika null = ganti menjadi "Tidak disebutkan")
            - Status: {is_email_sent} (jika 1 = ganti menjadi "Sudah Terkirim", jika 0 = ganti menjadi "Belum Terkirim")
            Petunjuk untuk Pengguna:
            - Buka Aplikasi Gmail di HP atau Melalui Browser pada Laptop/Desktop Anda.
            - Pastikan sudah masuk/login menggunakan akun google dari Undiksha.
            - Di Gmail, silahkan cek email Anda dari e-Ganesha Undiksha.
            - Silahkan tekan "Klik Untuk Reset Password".
            - Ikuti langkah untuk memasukkan password baru yang sesuai.
            - Jika sudah berhasil, silahkan login kembali ke SSO Undiksha.
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
        state["finishedAgents"].add("resetAccount_agent") 

        return {"answerAgents": [agentOpinion]}

    except Exception as e:
        # print("Error retrieving account information:", e)
        prompt = f"""
            Anda adalah seorang pengirim pesan informasi Undiksha.
            Tugas Anda untuk memberitahu pengguna bahwa:
            Pengajuan proses reset password akun SSO Undiksha tidak berhasil.
            - Ini pesan kesalahan dari sistem coba untuk diulas lebih lanjut agar lebih sederhana untuk diberikan ke pengguna: {reset_sso_info}
        """
        messages = [
            SystemMessage(content=prompt)
        ]
        response = chat_llm(messages)

        agentOpinion = {
            "question": state["accountQuestion"],
            "answer": response
        }
        state["finishedAgents"].add("resetAccount_agent") 
        
        return {"answerAgents": [agentOpinion]}