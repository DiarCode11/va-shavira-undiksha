from langchain_core.messages import HumanMessage, SystemMessage
from utils.agent_state import AgentState
from utils.llm import chat_llm
from utils.debug_time import time_check


@time_check
def resultWriterAgent(state: AgentState):
    if len(state["finishedAgents"]) < state["totalAgents"]:
        print("\nMenunggu agent lain menyelesaikan tugas...")
        return None

    elif len(state["finishedAgents"]) == state["totalAgents"]:
        info = "\n--- RESULT WRITER ---"
        print(info)

        prompt = f"""
            Berikut pedoman yang harus diikuti untuk menulis ulang informasi:
            - Awali dengan "Salam Harmoni🙏"
            - Berikan informasi secara lengkap dan jelas apa adanya sesuai informasi yang diberikan.
            - Urutan informasi sesuai dengan urutan pertanyaan.
            - Jangan menyebut ulang pertanyaan secara eksplisit.
            - Jangan menjawab selain menggunakan informasi pada informasi yang diberikan, sampaikan dengan apa adanya jika Anda tidak mengetahui jawabannya.
            - Jangan tawarkan informasi lainnya selain informasi yang diberikan yang didapat saja.
            - Terakhir tampilkan flag sumber hanya jika ada.
            - Hasilkan response dalam format Markdown.
            Berikut adalah informasinya:
            {state["answerAgents"]}
        """

        messages = [
            SystemMessage(content=prompt),
            HumanMessage(content=state["question"])
        ]
        response = chat_llm(messages)
        
        state["responseFinal"] = response
        
        return {"responseFinal": state["responseFinal"]}