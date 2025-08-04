import streamlit as st
import gait as G
from guidedAnalysis import GuidedAnalysisSystem
from agents import expert_interact, arcgis_doc_interact, paperAgent
from gait import Agent
import os

# Initialize system
react_tools = GuidedAnalysisSystem(max_turns=50)

reason_agent = Agent(
    model=f"azure/{os.environ['MODEL']}",
    base_url=f"{os.environ['AZURE_API_URL']}/{os.environ['MODEL']}",
    api_version=os.environ['AZURE_API_VERSION'],
    max_tokens=20000,
)

# Scratchpad for state
if 'sp' not in st.session_state:
    sp = G.ScratchpadInMemory()
    sp["history"] = react_tools.start_session("i want to find spatial clusters in hospital data")
    sp["user_information"] = [{'role': 'user', 'content': "i want to find spatial clusters in hospital data"}]
    sp["turn"] = 0
    sp["finished"] = False
    st.session_state.sp = sp

sp = st.session_state.sp

st.title("Spatial Clustering Co-Scientist")
st.markdown("This is an interactive agent guiding you through spatial clustering analysis.")

# Reasoning step
if not sp.get("last_action") and not sp.get("finished"):
    agent_reply = reason_agent(sp["history"]).content
    action, input_query = react_tools.react_decision(agent_reply)
    sp["input"] = input_query
    sp["last_action"] = action
    sp["history"].append({'role': 'assistant', 'content': agent_reply})

# Display agent reasoning
if sp.get("last_action") == "ask_user":
    question_prompt = react_tools.set_ask_user_prompt(
        f"History information: {sp['history'][:-1]}\nThe Latest observation: {sp.get('observation', '')}"
    )
    agent_reply = reason_agent(question_prompt).content
    st.markdown("**Agent's Question:**")
    st.info(agent_reply)
    sp["history"].append({'role': 'agent', 'content': agent_reply})

    user_input = st.text_input("Your answer or clarification:", key="user_input")
    if st.button("Submit Response") and user_input:
        sp["history"].append({'role': 'user', 'content': user_input})
        sp["last_action"] = None
        st.rerun()

elif sp.get("last_action") == "talk_expert":
    sp["observation"] = expert_interact(sp["input"])
    sp["history"].append({'role': 'system', 'content': sp["observation"]})
    sp["last_action"] = None
    st.rerun()

elif sp.get("last_action") == "arcgis_pro_document_retrieval":
    sp["observation"] = arcgis_doc_interact(sp["input"])
    sp["history"].append({'role': 'system', 'content': sp["observation"]})
    sp["last_action"] = None
    st.rerun()

elif sp.get("last_action") == "literature_search":
    sp["observation"] = paperAgent(sp["input"]["goal"], sp["input"]["keywords"])
    sp["history"].append({'role': 'system', 'content': sp["observation"]})
    sp["last_action"] = None
    st.rerun()

elif sp.get("last_action") == "finish":
    sp["finished"] = True
    st.success("✅ Analysis complete.")
    with st.expander("See full conversation history"):
        for turn in sp["history"]:
            st.markdown(f"**{turn['role'].capitalize()}**: {turn['content']}")
