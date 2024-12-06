from utils.agent_state import AgentState
from utils.debug_time import time_check


@time_check
def anomalyAccountAgent(state: AgentState):
    info = "\n--- Anomaly Account ---"
    print(info)

    prompt = f"""
        Mohon maaf, saya tidak dapat membantu menangani akun SSO atau Google Undiksha Anda.
        Berikut petunjuk untuk disampaikan kepada pengguna berdasarkan informasi dari akun pengguna:
        - Silahkan datang langsung ke Kantor UPA TIK Undiksha untuk mengurus akun Anda.
        - Atau cek pada kontak kami berikut: https://upttik.undiksha.ac.id/kontak-kami
    """

    agentOpinion = {
        "question": state["accountQuestion"],
        "answer": prompt
    }
    state["finishedAgents"].add("anomalyAccount_agent")
    
    return {"answerAgents": [agentOpinion]}
