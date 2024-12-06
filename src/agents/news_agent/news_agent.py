from langchain_core.messages import HumanMessage, SystemMessage
from utils.agent_state import AgentState
from utils.llm import chat_llm
from utils.debug_time import time_check
from utils.scrapper_rss import scrap_news


@time_check
def newsAgent(state: AgentState):
    info = "\n--- NEWS ---"
    print(info)

    result = scrap_news()
    state["newsScrapper"] = result

    prompt = f"""
        Anda adalah seorang pengelola berita.
        Berikut berita yang terbaru saat ini.
        - Data Berita: {state["newsScrapper"]}
    """

    messages = [
        SystemMessage(content=prompt),
        HumanMessage(content=state["newsQuestion"])
    ]
    response = chat_llm(messages)
    
    agentOpinion = {
        "answer": response
    }
    state["finishedAgents"].add("news_agent")
    
    return {"answerAgents": [agentOpinion]}
