import re
import dspy
from tools.expert_interact import rag_expert_input_fn

def split_sections(text):
    """
    自动按 section 标题（##）切分为list of dict: [{"title":..., "content":...}]
    """
    section_pattern = r"(## [0-9.]+ .+)"
    matches = list(re.finditer(section_pattern, text))
    sections = []
    for i, match in enumerate(matches):
        start = match.start()
        end = matches[i+1].start() if i+1 < len(matches) else len(text)
        title = match.group(1).strip()
        content = text[start+len(title):end].strip()
        sections.append({"title": title, "content": content})
    return sections


def llm_sections_need_expert(sections, history, llm_engine):
    """
    Use LLM to determine whether expert intervention is recommended for each section and return a boolean list.
    """
    results = []
    for sec in sections:
        prompt = f"""Here is the history of the conversation:
        {history}
        You are an expert assistant. Decide if the following section of a hypothesis proposal would benefit from domain expert review (e.g., if it involves contextual judgment, unclear assumptions, or choices that may affect downstream validity). 
        Return "YES" if you recommend expert input, "NO" if not.
        Section Title: {sec['title']}
        Section Content:
        {sec['content']}
        ----
        Reply YES or NO."""

        output,_,_ = llm_engine.respond([{'role': 'user','content': prompt}])
        flag = "YES" in output.strip().upper()
        results.append(flag)
    return results

def expert_review_sections(sections, need_expert_flags, expert_input_fn=None):
    """
    Only for the "need_expert" part,
    a pop-up window is displayed to collect expert suggestions.
    For the remaining parts, they are directly processed through.
    """
    revised_sections = []
    for idx, sec in enumerate(sections):
        if need_expert_flags[idx]:
            print(f"\n=== [Expert intervention] {sec['title']} ===")
            print(sec['content'])
            if expert_input_fn:
                feedback = expert_input_fn(sec['title'], sec['content'])
            else:
                feedback = input("请专家批注/建议修改（直接回车为通过）：\n")
            if feedback.strip():
                sec['content'] += f"\n\n【专家修订】\n{feedback}"
        revised_sections.append(sec)
    return revised_sections

def merge_sections_to_text(sections):
    """
    convert the list of sections back to a single text string.
    """
    text = ""
    for sec in sections:
        text += f"{sec['title']}\n\n{sec['content']}\n\n---\n"
    return text

def expert_interact_workflow(agent_output, history, llm_engine, expert_input_fn=None):
    # 1. split
    sections = split_sections(agent_output)
    # 2. judge whether expert intervention is needed
    need_expert_flags = llm_sections_need_expert(sections, llm_engine)
    # 3. expert interaction
    revised_sections = expert_review_sections(sections, need_expert_flags, expert_input_fn)
    # 4. merge the sections back to text
    merged = merge_sections_to_text(revised_sections)
    return merged

if __name__ == "__main__":
    with open("hypothesis.txt", "r") as f:
        text = f.read()

    print(split_sections(text))