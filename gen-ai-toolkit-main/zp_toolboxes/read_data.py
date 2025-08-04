import json
import tiktoken
def generate_llm_context_from_layer_json(json_path, output_path=None, max_values=1):
    with open(json_path, "r", encoding="utf-8") as f: data = json.load(f)
    all_contexts = []
    for layer in data.get("layers", []):
        name = layer.get("name", "Unknown Layer")
        stype = layer.get("stype", "Unknown Type")
        uri = layer.get("uri", "Unknown URI")
        context = f'Layer: "{name}"; Type: {stype}; File Path: "{uri}"; This layer represents {name.lower()} with the following attributes: '
        col_descriptions = []
        for col in layer.get("columns", []):
            cname = col.get("name", "Unknown")
            alias = col.get("alias", "")
            dtype = col.get("dtype", "Unknown")
            values = col.get("values", [])[:max_values]
            vstr = ", ".join(f'"{v}"' for v in values)
            desc = f'{cname} ({dtype})'
            if alias and alias.lower() != cname.lower(): desc += f' aka "{alias}"'
            if values: desc += f': Examples: {vstr}'
            col_descriptions.append(desc)
        context += " | ".join(col_descriptions)
        all_contexts.append(context)
    final_context = " || ".join(all_contexts)
    if output_path:
        with open(output_path, "w", encoding="utf-8") as f: f.write(final_context)
    encoding = tiktoken.encoding_for_model("gpt-4")
    tokens = encoding.encode(final_context)
    # print(len(tokens))

    return final_context
if __name__ == "__main__":
    ctx = generate_llm_context_from_layer_json(r"C:\Users\64963\OneDrive - The University of Texas at Austin\PhD\6-Job\ESRI\Spatial_Co_Scientist\gen-ai-toolkit-main\temp_folder\data_context.json")

    print(ctx)
