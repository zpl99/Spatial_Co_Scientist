import os.path

from openai import OpenAI, APIConnectionError, APITimeoutError, RateLimitError, InternalServerError, AzureOpenAI
from glob import glob

engine = AzureOpenAI(
    api_key="3c744295edda4f13bf0ef198ddb4d24c",
    api_version="2024-10-21",
    azure_endpoint="https://ist-apim-aoai.azure-api.net/load-balancing/gpt-4.1"
)

def call_llm_chunk(transcript_chunk, history):
    prompt = f"""You are an expert assistant for analyzing meeting transcripts.

Your workflow is as follows:

1. You are processing a long transcript in separate parts (chunks). You will always see the previously extracted Q&A history, and a new transcript chunk. Use the history to ensure continuity and avoid duplicate extraction.
2. Cynthia M Sims (often called "Sims" in the transcript) is always the one who asks questions. Each time Sims asks a question—**including small talk, greetings, check-ins, or casual questions**—it marks the start of a new Q&A session.
   - Ignore all lecture or statement content from Sims.
   - Whenever Sims asks a question (even if it is for greetings, emotional check-in, or small talk), **record her entire question as she phrased it**, including background, follow-up, or context, but only remove obvious filler words (such as "um", "well", "so", "you know", "like", etc.). Do not oversimplify. Keep her question as detailed and complete as possible, whether it is formal or informal.
3. After Sims asks a question, **all responses by other people before Sims next speaks** (regardless of how many speakers or how many turns) should be considered as answers to that question.
   - If the same responder speaks multiple times before Sims next speaks, combine all their responses into one paragraph (in the order they appear).
   - Do not repeat the responder's name for each utterance; use their name once and combine all content.
   - Keep the original answer content as much as possible, including explanation, context, and extra comments, but remove filler words.
   - Do not oversimplify or cut out important details. If a response is long or includes storytelling or background, keep it.
   - Different responders' answers should be kept as separate paragraphs.
4. When Sims speaks again (whether a new question, a comment, or something else), consider the previous Q&A session closed, and start a new session if she asks another question (including small talk).
5. Do not output Sims’s non-question content, narration, or other lecture content.
6. **Before outputting any question or answer, carefully check the Q&A history below and make sure you are NOT repeating any question or answer that already exists in the history. Only output genuinely new or newly completed content. Do not include duplicates.**
7. For each question and its answers, use this plain text format (NOT JSON):

Cynthia M Sims ：{{full, detailed question as spoken}}
{{responder name}}: {{that responder’s complete, combined answer}}
{{another responder name (if applicable)}}: {{their complete, combined answer}}

(Repeat for each question and its answers. Only output what is new or what is newly completed with this chunk. Do not leave out background or details from the original speech, whether formal or informal.)

Below is the previously extracted Q&A history:
-----
{history}
-----

Below is the new transcript chunk:
-----
{transcript_chunk}
-----

Here is the example: 
Cynthia M. Sims: So—three words, how are you doing?
Pipiet Larasatie: I can start. Tired but excited. So many things are happening—we are trying to purchase a home. It's actually like a blessing in disguise or something like that. We just moved to this area four months ago, and we were surprised that the housing prices are crazy. We thought we couldn’t afford anything. But then everything kind of felt like it was already prepared—like things just flowed in a way beyond my power.
We met a fabulous real estate agent, and then we met a loan officer. They told us that, since we're here under a visa, we need to move very quickly. The new administration no longer offers FHA, USDA, or first-time homebuyer loans to non-residents. So I only had less than two weeks to get something signed with a seller. Those two weeks were hectic, but we got something. Now we're waiting for the underwriting process.
Also, right now I'm writing a grant—trying to apply for something with USDA. They’ve frozen almost everything except this specific program, and the deadline is tomorrow. So I’m tired, but I’m excited.

Output ONLY in the above plain text format. Do not add any explanations or extra information.

"""
    response = engine.chat.completions.create(
        model="gpt-4.1",
        max_tokens=32768,
        messages=[{"role": "user", "content": prompt}]
    )
    return response.choices[0].message.content.strip()

def process_large_transcript(vtt_file, chunk_size=150):
    with open(vtt_file, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    chunks = ["".join(lines[i:i+chunk_size]) for i in range(0, len(lines), chunk_size)]
    all_txt = []
    history_str = ""
    for idx, chunk in enumerate(chunks):
        print(f"Processing chunk {idx+1}/{len(chunks)}")
        llm_out = call_llm_chunk(chunk, history_str)
        if llm_out.strip():
            all_txt.append(llm_out)
            history_str += "\n" + llm_out + "\n"
    return "\n\n".join(all_txt)

if __name__ == "__main__":
    allfiles = glob("/Users/zepingliu/Desktop/transcriptions/*.vtt")
    for file in allfiles:
        result = process_large_transcript(file)
        with open(os.path.basename(file).split(".")[0]+"txt", "w", encoding="utf-8") as f:
            f.write(result)
