import streamlit as st
from guidedAnalysis import GuidedAnalysisAgent, LLMEngine

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
        for turn in st.session_state.history:
            if turn['role'] == 'user':
                st.markdown(f"**You:** {turn['content']}")
            elif turn['role'] == 'assistant':
                st.markdown(f"<span style='color:blue'>**Agent:** {turn['content']}</span>", unsafe_allow_html=True)
            elif turn['role'] == 'system':
                st.markdown(f"<span style='color:green'>*{turn['content']}*</span>", unsafe_allow_html=True)

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
