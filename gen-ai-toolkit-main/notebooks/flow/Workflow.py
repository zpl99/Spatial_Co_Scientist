#%%
import gait as G
from IPython.display import Markdown, display
#%%
class GenerateOutline(G.Node):
    def exec(
        self,
        sp: G.Scratchpad,
    ) -> str:
        topic = sp["topic"]
        agent = G.Agent(model="ollama_chat/gemma3:4b", temperature=0.2)
        resp = agent(f"Create a detailed outline for an article about {topic}")
        sp["outline"] = resp.content
        return G.Node.DEFAULT
#%%
class WriteContent(G.Node):
    def exec(
        self,
        sp: G.Scratchpad,
    ) -> str:
        outline = sp["outline"]
        agent = G.Agent(model="ollama_chat/gemma3:4b", temperature=0.2)
        resp = agent(f"Write content based on this outline: {outline}")
        sp["content"] = resp.content
        return G.Node.DEFAULT
#%%
class Review(G.Node):
    def exec(
        self,
        sp: G.Scratchpad,
    ) -> str:
        content = sp["content"]
        agent = G.Agent(model="ollama_chat/gemma3:4b", temperature=0.2)
        resp = agent(f"Review and improve this draft: {content}")
        sp["review"] = resp.content
        return G.Node.DEFAULT
#%%
(head := GenerateOutline()) >> WriteContent() >> Review()

flow = G.Flow(head)

flow.display_markdown()
#%%
flow(topic="AI Topic") # Any argument, is placed based on the argname in the scratchpad.
#%%
display(Markdown(flow["outline"]))
#%%
display(Markdown(flow["content"]))
#%%
display(Markdown(flow["review"]))