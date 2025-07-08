import streamlit as st
from guidedAnalysis import GuidedAnalysisAgent, LLMEngine
import re
def get_user_visible_message(agent_reply, action, ask_user_flag, observation=None):
    """
    Generate a user-friendly message for display based on the current agent action.
    See detailed docstring in previous reply for behavior.
    """
    if ask_user_flag:
        # If the agent wants a clarification, display the question directly.
        # Extract Thought and Ask User parts
        thought_match = re.search(r'Thought:(.*?)(?:\n\n|$)', agent_reply, re.DOTALL)
        action_match = re.search(r'Action: ask_user; Input:\s*"(.*)"', agent_reply)

        thought = thought_match.group(1).strip() if thought_match else ""
        ask_user = action_match.group(1).strip() if action_match else ""

        # Display with colors in Streamlit
        # if thought:
        #     st.markdown(
        #         f"<span style='color:#2563eb; font-weight:bold'>System Thought (debug only):</span> <span style='color:#1e40af'>{thought}</span>",
        #         unsafe_allow_html=True)
        if ask_user:
            st.markdown(
                f"<span style='color:#059669; font-weight:bold'>Ask User:</span> <span style='color:#047857'>{ask_user}</span>",
                unsafe_allow_html=True)
        return agent_reply.strip()

    if action == "finish":
        # If analysis is done, display closing statement.
        return agent_reply.strip()

    if action == "literature_search":
        tip = "🔍 Reviewing relevant academic literature for your request."
        if observation:
            if isinstance(observation, dict) and "articles_with_reasoning" in observation:
                summary = observation["articles_with_reasoning"]
                if len(summary) > 3000:
                    summary = summary[:3000] + "..."
                tip += "\n\n**Summary:**\n" + summary
            elif isinstance(observation, str):
                tip += "\n\n**Summary:**\n" + (observation[:3000] + "..." if len(observation) > 3000 else observation)
        st.markdown(f"<span style='color:green'>{tip}</span>", unsafe_allow_html=True)
        return tip

    if action == "expert_knowledge_interact":
        tip = "🧑‍🔬 Consulting domain expert knowledge."
        if observation:
            if isinstance(observation, str) and len(observation) > 0:
                summary = observation[:10000] + ("..." if len(observation) > 10000 else "")
                tip += "\n\n**Expert insight:**\n" + summary
        st.markdown(f"<span style='color:green'>{tip}</span>", unsafe_allow_html=True)
        return tip

    if action == "arcgis_document_retrieval":
        tip = "📖 Looking up ArcGIS Pro documentation."
        if observation:
            summary = observation[:10000] + ("..." if len(observation) > 10000 else "")
            tip += "\n\n**Documentation result:**\n" + summary
        st.markdown(f"<span style='color:green'>{tip}</span>", unsafe_allow_html=True)
        return tip

    if action == "unknown action" or action is None:
        st.markdown(f"<span style='color:red'>The system has received your input and is processing the next step...</span>", unsafe_allow_html=True)
        return "The system has received your input and is processing the next step..."

    return None

if 'history' not in st.session_state:
    st.session_state.history = []
if 'awaiting_user' not in st.session_state:
    st.session_state.awaiting_user = False
if 'finished' not in st.session_state:
    st.session_state.finished = False

st.title("Spatial Co-Scientist: Guided Analysis Demo")

def restart():
    st.session_state.clear()
    st.rerun()

if st.session_state.history == [] and not st.session_state.finished:
    user_input = st.text_input("Describe your spatial analysis need (e.g. 'I want to find retail clusters')", key="init_input")
    if user_input:
        llm_engine = LLMEngine("gpt-4.1")
        st.session_state.agent = GuidedAnalysisAgent(llm_engine, max_turns=10)
        st.session_state.history = st.session_state.agent.start_session(user_input)
        st.session_state.awaiting_user = False
        st.rerun()   # 确保页面刷新到chat状态

if st.session_state.finished:
    st.success("✅ Analysis complete! If you want to start over, please refresh or click below.")
    st.button("Restart", on_click=restart)
else:
    if st.session_state.history:
        if not st.session_state.awaiting_user:
            # Agent turn
            agent = st.session_state.agent
            while True:
                agent_reply = agent.next_turn(st.session_state.history)
                action, observation, ask_user_flag = agent.react_decision(agent_reply)
                st.session_state.history.append({'role': 'assistant', 'content': agent_reply})

                if action and not ask_user_flag and action != "finish":
                    obs_msg = f"Observation: {observation}"
                    st.session_state.history.append({'role': 'system', 'content': obs_msg})
                    continue
                if ask_user_flag or not action:
                    st.session_state.awaiting_user = ask_user_flag or not action
                    break
                if action == "finish":
                    st.session_state.awaiting_user = False
                    st.session_state.finished = True
                    break

        # Show chat
        for i, turn in enumerate(st.session_state.history):
            if turn['role'] == 'user':
                st.markdown(f"**You:** {turn['content']}")
            elif turn['role'] == 'assistant':
                # Try to extract relevant message for user
                # Peek ahead for the related observation/system turn
                action, observation, ask_user_flag = None, None, False
                # Try to find a paired observation
                if i + 1 < len(st.session_state.history) and st.session_state.history[i + 1]['role'] == 'system':
                    observation = st.session_state.history[i + 1]['content'].replace("Observation: ", "", 1)
                # Use react_decision to recover action and ask_user_flag for this turn
                agent = st.session_state.agent
                action, _, ask_user_flag = agent.react_decision(turn['content'])
                user_msg = get_user_visible_message(turn['content'], action, ask_user_flag, observation)
                # if user_msg:
                #     st.markdown(f"<span style='color:blue'>**Agent:** <br>{user_msg}</span>", unsafe_allow_html=True)
            # Optionally, skip 'system' turns completely (since their content is embedded above)

        # User input
        if st.session_state.awaiting_user:
            # 用一个唯一key（比如history长度）
            box_key = f"userinput_{len(st.session_state.history)}"
            user_input = st.text_input("Your answer or clarification:", key=box_key)
            if user_input:
                st.session_state.history.append({'role': 'user', 'content': user_input})
                st.session_state.awaiting_user = False
                # 清空输入框
                st.rerun()