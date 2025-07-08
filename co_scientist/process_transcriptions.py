import os
from glob import glob
from openai import AzureOpenAI

engine = AzureOpenAI(
    api_key="3c744295edda4f13bf0ef198ddb4d24c",
    api_version="2024-10-21",
    azure_endpoint="https://ist-apim-aoai.azure-api.net/load-balancing/gpt-4.1"
)

def call_llm_chunk(transcript_chunk, history):
    prompt = f"""
You are a transcription-cleaning assistant.

Your job is to clean and merge a meeting transcript, which may be provided in chunks.

**Your rules:**
- Remove all timestamps and numbering.
- Only keep lines in the format: "Speaker: content".
- If the same speaker speaks multiple times in a row (even across chunks), **combine all of their speech into a single block**. Only start a new speaker block when a different person speaks.
- Do not break up or restart a speaker's block if there is no interruption by another speaker, even if the transcript is chunked.
- Do not change any speaker names or actual dialog content except for removing filler words.
- Output only the cleaned, merged dialog in this format (no extra explanation):

Speaker A: all their merged content here (until another person speaks)
Speaker B: their merged content here (until another person speaks)
Speaker A: if A speaks again after B, continue here
...

**History:**
Below is the cleaned transcript from the previous chunks (history).  
- If the last speaker in the history is also the first speaker in the new chunk, continue their block, merging the dialog seamlessly.
- Do not duplicate or split dialog unnecessarily.

-----
History:
{history}
-----

Below is the new transcript chunk to process:
-----
{transcript_chunk}
-----

Your job:
1. Clean the new chunk as above.
2. Merge it with the history, combining the same speaker's dialog if needed.
3. Output the complete merged transcript in the specified format.  
4. Do NOT add any explanations or extra comments.
5. Think step by step
"""

    response = engine.chat.completions.create(
        model="gpt-4.1",
        max_tokens=30000,
        messages=[{"role": "user", "content": prompt}]
    )
    return response.choices[0].message.content.strip()

def process_large_transcript(vtt_file, chunk_size=300):
    with open(vtt_file, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    chunks = ["".join(lines[i:i+chunk_size]) for i in range(0, len(lines), chunk_size)]
    history_str = ""
    for idx, chunk in enumerate(chunks):
        print(f"Processing chunk {idx+1}/{len(chunks)}")
        llm_out = call_llm_chunk(chunk, history_str)
        print(llm_out)
        # 只保留最新的完整history（LLM会输出整个合并后的对话文本）
        history_str = llm_out
    return history_str

if __name__ == "__main__":
    allfiles = glob("/Users/zepingliu/Desktop/result/*.vtt")
    for file in allfiles:
        result = process_large_transcript(file)
        outname = os.path.basename(file).rsplit(".", 1)[0] + ".txt"
        with open(outname, "w", encoding="utf-8") as f:
            f.write(result)
