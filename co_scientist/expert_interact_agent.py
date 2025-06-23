import re
from sys import flags
import json
import dspy
from tools.expert_interact import rag_expert_input_fn

def split_sections(text):
    """
    split the text into sections based on the pattern of headings.
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

def json_to_title_content_list(data):
    data = json.loads(data)
    result = []

    # 处理单条字符串的键值对
    def add_entry(title, content):
        if isinstance(content, list):
            # 列表转成多行字符串
            content_str = "\n".join(str(item) for item in content)
        elif isinstance(content, dict):
            # dict直接转json字符串
            import json
            content_str = json.dumps(content, ensure_ascii=False, indent=2)
        else:
            content_str = str(content)
        result.append({"title": title, "content": content_str})

    # 针对你给的结构，逐个字段处理
    # 这些字段都可以当成一个条目
    for key in data:
        add_entry(key, data[key])

    return result

def llm_sections_need_expert(sections, history, llm_engine, n_recent_history=5):
    """
    sections: List[str], 每一节文本
    history: List[dict], [{role, content}], 全局历史
    llm_engine: 你的LLM对象，有 respond(user_input, ...)
    return: List[bool]
    """
    flags = []
    # short history excerpt for context
    hist_excerpt = "\n".join([f"[{h['role']}]: {h['content']}" for h in [history[0], history[-1]]])
    for section in sections:
        prompt = f"""You are assisting in scientific hypothesis writing. Below is a section from the draft and the conversation history. 

        Your job: decide if this section requires review or modification by a human domain expert (e.g., because it contains ambiguous reasoning, lacks expert-specific context, uses uncertain language, or is methodologically controversial).
        
        Return only "True" if it should be reviewed by an expert, otherwise return "False".
        
        ## Section:
        {section}
        
        ## Recent Conversation History:
        {hist_excerpt}
        
        Does this section require expert intervention? Return only "True" or "False".
        """
        user_input = [{"role": "user", "content": prompt}]
        output, *_ = llm_engine.respond(user_input, temperature=0, top_p=1)
        flags.append("true" in output.lower())
    return flags

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
                feedback = input("Ask experts to make remarks/suggestions for modification. (Press Enter to pass.)\n")
            if feedback.strip():
                sec['content'] += f"\n\nExpert edit: \n{feedback}"
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
    sections = json_to_title_content_list(agent_output)
    # 2. judge whether expert intervention is needed
    # need_expert_flags = llm_sections_need_expert(sections, history, llm_engine)
    need_expert_flags = [False, True, True, True, True, True, False]  # For demo, assume all sections need expert review
    # 3. expert interaction
    revised_sections = expert_review_sections(sections, need_expert_flags, expert_input_fn)
    # 4. merge the sections back to text
    merged = merge_sections_to_text(revised_sections)
    return merged

if __name__ == "__main__":
    with open("hypothesis.txt", "r") as f:
        text = f.read()

    print(split_sections(text))