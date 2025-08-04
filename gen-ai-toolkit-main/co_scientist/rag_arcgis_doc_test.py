# newAPI.py

import logging
from time import sleep
from typing import Any, Dict, List, Optional
from datetime import datetime, timedelta
import json
import os
import dotenv
import re
from httpx import Client, Response
import openai

logger = logging.getLogger(__name__)

dotenv.load_dotenv()

core_concepts = [
    {
        "name": "Location",
        "description": "Spatial information is always linked to location in some way, primarily to answer where questions. Perhaps counter-intuitively, location is a relation, not a property. Nothing has an intrinsic location, even if it always remains where it is. All location descriptions express spatial relations between figures to be located and chosen grounds (a region, a street network, coordinate axes). How one locates things, that is, what ground and what spatial relation one chooses, depends on context. When grounds become salient, as in the case of places, they tend to be thought of as ‘locations’ in the sense of objects. Spatial reference systems standardize location relations and turn them into attributes, describing positions in a system. Yet, when data use multiple reference systems (e.g. latitude and longitude as well as projected coordinates), locations need to be understood as relations and interpreted with respect to their grounds (e.g. the Greenwich meridian and equator)."
    },
    {
        "name": "Neighbourhood",
        "description": "Relating different phenomena through location is fundamental to spatial analysis. The great power of such locational analyses results from the fact that nearby things tend to be more related than distant things (Tobler 1970). Nearness, or rather the neighbourhood answering the question what is near, is therefore a natural companion concept to location. Neighbourhoods are commonly thought of as regions, characterizing the spatial context. Definitions of near and neighbourhood are not only context dependent, but also necessarily vague. Even if the context to be captured is specified (e.g. the region from which one can walk to a bus station), neighbourhoods remain imprecisely defined and lack crisp boundaries."
    },
    {
        "name": "Field",
        "description": "Fields describe phenomena that have a scalar or vector attribute everywhere in a space of interest, for example, air temperatures on the Earth’s surface. Field information answers the question what is here?, where here can be anywhere in the space considered. Generalizing the field notion from physics, field-based spatial information can also represent attributes that are computed rather than measured, such as probabilities or densities."
    },
    {
        "name": "Object",
        "description": "Objects describe individuals that have an identity as well as spatial, temporal, and thematic properties. Object information answers questions about properties and relations of objects. It results from fixing theme, controlling time, and measuring space. Features, such as surfaces or parts of them, depend on objects, but can also be understood as a special case of them. The notion of an object implies boundedness, but this does not mean that the object’s boundaries need to be known or even knowable, only that there are limits outside of which there are no parts of the object. Crude examples of such limits are the minimal bounding boxes used for indexing and querying objects in databases. Many objects (particularly natural ones) do not have crisp boundaries (Burrough and Frank 1996). Differences between spatial information from multiple sources are often caused by more or less arbitrary delimitations through context-dependent boundaries. Many questions about objects and features can be answered without boundaries, using simple point representations with thematic attributes."
    },
    {
        "name": "Network",
        "description": "Network information answers questions about connectivity, which is central to space and spatial information. It captures binary relationships among arbitrary numbers of objects, called the nodes or vertices of a network. Any relation of interest can connect the nodes and be represented by edges. The spatiality of a network results from positioning the nodes in some space and may involve geometric properties of the edges, such as their length or shape. The two main kinds of networks encountered in spatial information are link and path networks. Link networks capture logical or other abstract relationships between nodes, such as friendships, business relations, or treaties between social agents. Path or transportation networks model systems of paths along which matter, energy, or information flows. Examples are roads, utilities, communication lines, synapses, blood vessels, or electric circuits. Network applications benefit from the well-studied representations of networks as graphs and the correspondingly vast choice of algorithms and data structures. Partly due to this sound mathematical and computational basis, networks are the spatial concept that is most broadly recognized and applied across disciplines. One may speculate from this success story that a similar level of understanding and formalization of the other core concepts will encourage their use in transdisciplinary work. As the exposure here shows, such a level has not yet been reached in most cases."
    },
    {
        "name": "Event",
        "description": "Events and processes are of central interest to science and society, for answering questions about change. Spatial events manifest themselves through changes of locations (i.e. motion), neighbourhoods, fields, objects, and networks, that is, the changes to instances of the previous core concepts. Events get related through temporal relations as well as through spatial relations among their participants. They can be seen as carved out of processes in the same way that physical objects are carved out of matter, that is, by bounding the processes and giving each event an identity"
    },
    {
        "name": "Granularity",
        "description": "Granularity is the first (and most spatial) concept of information on the list. It characterizes the size of the spatial, temporal, and thematic units about which information is reported. Granularity information answers questions about the precision of spatial information. It matters most when taking and evaluating decisions based on that information. Granularity characterizes information about all concepts introduced so far: location is recorded at certain granularities, neighbourhoods can be identified at several levels, fields are recorded at certain spacings or sizes of cells, and the choice of the types of objects (say, buildings vs. cities) or nodes (say, transistors vs. people) determines the spatial granularities of object and network information. Events are defined and distinguished by choosing granularity levels in space, time, and theme."
    },
    {
        "name": "Accuracy",
        "description": "Accuracy is a key property of spatial information, capturing how information relates to the world. Information about accuracy answers questions about the correctness of spatial information. Assessing the accuracy of information requires two assumptions: that there is, at least in principle, correct information and that the results of repeated measurements or calculations distribute in some form regularly around it. The first assumption requires an unambiguous specification of the reported phenomenon and of the procedure to measure it. The second assumption requires an understanding of measurement as a random process."
    },
    {
        "name": "Meaning",
        "description": "Understanding what producers meant by some spatial information is crucial to its adequate use. Information about meaning (semantic information) answers the question how to interpret the terms used in spatial information. It concerns the spatial, temporal, and thematic components. Data and computations do not have a well-defined meaning by themselves, but are used by somebody to mean something in some context. Therefore, it is impossible to fix the meaning of terms in information. However, one can make the conditions for using and interpreting a term explicit. This is what ontologies do – they state constraints on the use and interpretation of terms. But language use is flexible and does not always follow rules, even for technical terms. An empirical account of how some terms are actually used can therefore provide additional insights on intended meaning and actual interpretation. This is what folksonomies deliver they list and group the terms with which information resources have been tagged."
    },
    {
        "name": "Value",
        "description": "The final core concept proposed is that of value. Information about values attached to or affected by spatial information answers questions about the roles played by spatial information in society. The prototypical value is economic, but the valuation of spatial information as a good in society goes far beyond monetary considerations. It includes assessing the relation of spatial information to other important values in society, such as privacy, trust, infrastructure, or heritage. Given the wide-ranging aspects of spatial information values, no coherent theoretical framework for them can be expected any time soon. Even theories about the economic value of spatial information remain sketchy and difficult to apply, because they involve parameters that are hard to generalize, control, and measure. The values of information, economic and other, tend to accrue holistically and unpredictably, by new questions that can be asked and answered."
    }
]


def get_urls_for_tier(tier: str = "dev") -> tuple[str, str]:
    base_url_key = f"BASE_URL_{tier.upper()}"
    token_url_key = f"TOKEN_URL_{tier.upper()}"
    base_url = os.getenv(base_url_key)
    token_url = os.getenv(token_url_key)
    if not base_url or not token_url:
        raise ValueError(
            f"Missing environment variables for tier '{tier}'. Check {base_url_key} and {token_url_key} values in .env")
    return base_url, token_url


def get_fresh_token(tier: str = "dev") -> Dict[str, Any]:
    _, token_url = get_urls_for_tier(tier)

    # Use tier-specific credentials
    username = os.getenv(f"USERNAME_{tier.upper()}")
    password = os.getenv(f"PASSWORD_{tier.upper()}")

    if not username or not password:
        raise ValueError(
            f"Missing credentials for tier '{tier}'. Check USERNAME_{tier.upper()} and PASSWORD_{tier.upper()}")

    client = Client(timeout=10)

    data = {
        "username": username,
        "password": password,
        "client": "referer",
        "referer": "https://aiassistant.mapsdevext.arcgis.com/",
        "expiration": 20160,
        "f": "pjson"
    }

    response = client.post(token_url, data=data)
    response.raise_for_status()
    print(response.json())
    return response.json()


def get_token(tier: str = "dev") -> str:
    token_file = f"auth_token_{tier}.json"
    if os.path.exists(token_file):
        with open(token_file, 'r') as f:
            token_data = json.load(f)
        expires = datetime.fromtimestamp(token_data['expires'] / 1000)
        if datetime.now() < expires:
            return token_data['token']
    # Couldn't find the file, or the token has expired
    token_data = get_fresh_token(tier)
    with open(token_file, 'w') as f:
        json.dump(token_data, f)
    return token_data['token']


def simple_context_asserter(
        response: dict, context_kind: set[str], context_additional_fields: List[str] = []
):
    if response["context"]:
        assert response["context"]["kind"] in context_kind
        for keys in response["context"]:
            assert keys in context_additional_fields or keys == "kind"


def get_skill_ids(client: Client, auth_token: str) -> Dict[str, str]:
    """
    Helper method to get a dictionary where the key is the skill name and the value is the skill ID.
    :param client: The test client.
    :param auth_token: The auth token.
    :return: Dictionary mapping skill names to IDs.
    """
    url: str = "/skills"
    response: Response = client.get(url, headers={"token": f"{auth_token}"})
    assert response.status_code == 200
    response_json: Any = response.json()
    result_dict: Dict[str, str] = {}
    for skill_json in response_json:
        skill_id: str = skill_json["id"]
        skill_name: str = skill_json["manifest"]["name"]
        result_dict[skill_name] = skill_id

    return result_dict


def chat_with_skill(
        client: Client,
        skill_id: Optional[str],
        message: str,
        auth_token: str,
        previous_conversation_id: Optional[str] = None,
        context: Any = None,
) -> List[Dict]:  # retrofitted for python 3.9.1
    """
    Helper method to automate the chat process with a skill.
    :param context: The request context that needs to be passed.
    :param previous_conversation_id: The previous conversation id.
    :param client: The test client.
    :param skill_id: Optional skill id.
    :param message: The message to send.
    :param message_type: The type of message.
    :param auth_token: The auth token.
    :return: List of chat responses.
    """
    if skill_id:
        url = f"/skills/{skill_id}/chat"
    else:
        url = "/chat"
    responses: List[Dict] = []

    if previous_conversation_id:
        chat_request = {
            "message": message,
            "context": context,
            "conversation_id": previous_conversation_id,
        }
    else:
        chat_request = {
            "message": message,
            "context": context,
        }
    resp: Response = client.post(
        url,
        json=chat_request,
        headers={"token": f"{auth_token}"},
    )
    if resp.status_code != 200:
        raise Exception(f"Failed to send message: {message}. Response: {resp.content}")
    chat_response = resp.json()
    assert chat_response["hasMore"]
    responses.append(chat_response)
    has_next = chat_response["hasMore"]
    conversation_id = chat_response["conversationId"]
    inquiry_id = chat_response["inquiryId"]
    ack_sequence_number = chat_response["sequenceNumber"]
    logger.info(f"Test client Reading from conversation id: {conversation_id}")
    sleep(.5)
    while has_next:
        poll_chat_request = _build_polling_request(
            # user_id="1234",
            conversation_id=conversation_id,
            ack_sequence_number=ack_sequence_number,
            inquiry_id=inquiry_id,
        )
        resp = client.post(
            url,
            json=poll_chat_request,
            headers={"token": f"{auth_token}"},
        )
        logger.info(f"Test client ackSequenceNumber: {ack_sequence_number}")
        if resp.status_code != 200:
            logger.error(
                "Failed to poll chat. If this persists, please contact the devs."
            )
            return None
        chat_response = resp.json()
        response_seq_num = chat_response["sequenceNumber"]
        logger.info(f"Test client sequenceNumber: {response_seq_num}")
        assert ack_sequence_number is not None

        responses.append(chat_response)
        ack_sequence_number = response_seq_num
        has_next = chat_response["hasMore"]
        sleep(1)
    return responses


def _build_polling_request(
        conversation_id: str, ack_sequence_number: str, inquiry_id: str
):
    return {
        "conversationId": conversation_id,
        "inquiryId": inquiry_id,
        "ackSequenceNumber": ack_sequence_number,
    }


def simple_answer(prompt: str, skill_id: str = "doc_chat", full_response: bool = False,
                  context={"kind": "DocAIAssistantRequest", "filters": {}}, tier: str = "dev"):
    token = get_token(tier)
    base_url, _ = get_urls_for_tier(tier)
    client = Client(base_url=base_url, timeout=60.0)
    try:
        responses = chat_with_skill(
            client=client, skill_id=skill_id, message=prompt, auth_token=token, context=context
        )
        if full_response:
            return responses
        return extract_response(responses, skill_id)

    except Exception as e:
        logger.error(f"Error in simple_answer: {e}")
        return f"Error accessing: {str(e)}"


def extract_response(responses: List[Dict], skill_id: str) -> str:
    if skill_id == "doc_chat":
        for interaction in responses:
            if interaction.get("message") is not None:
                return interaction["message"]
    elif skill_id == "doc_ai_assistant":
        last_reply = ""

        for interaction in responses:
            context = interaction.get("context")
            if context and context.get("kind") == "DocAIAssistantContext":
                results = context.get("results", [])
                if results and len(results) > 0:
                    reply = results[0].get("reply", "")
                    if reply:
                        last_reply = reply

        if not last_reply:
            return "Error accessing: No response"

        return last_reply
    else:
        return "Invalid skill ID"


def extract_toolset_names(response):
    """Robustly extract toolset names from response string, always return a list."""
    if not response or not isinstance(response, str):
        return []
    lines = response.splitlines()
    toolset_names = []
    for line in lines:
        line = line.strip()
        if line.startswith('- '):
            toolset_name = line[2:].strip()
            if toolset_name and toolset_name.lower() != "none":
                toolset_names.append(toolset_name)
    return list(dict.fromkeys(toolset_names))


def extract_tools(response):
    """Robustly extract tool names from a comma-separated string, always return a list."""
    if not response or not isinstance(response, str):
        return []
    tools = [t.strip() for t in response.split(",") if t.strip() and t.strip().lower() != "none"]
    return list(dict.fromkeys(tools))


def llm_supervisor(message):
    prompt = f"""Review the LLM output below to clean out the list of real and usable names and output only the final list of valid entries. An empty list is returned if the content is invalid.
Original content: {message}
The result is just a json array with no explanation. For example: ["toolset1", "toolset2"]
"""
    try:
        client = openai.AzureOpenAI(api_key="3c744295edda4f13bf0ef198ddb4d24c",
                                    api_version="2024-10-21",
                                    azure_endpoint="https://ist-apim-aoai.azure-api.net/load-balancing/gpt-4o")

        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": "You are a helpful assistant for filtering and validating tool lists."},
                {"role": "user", "content": prompt}
            ],
            temperature=0,
            max_tokens=1000
        )
        result_text = response.choices[0].message.content.strip()
        return json.loads(result_text)
    except Exception as e:
        print(f"Supervisor error: {e}")
        return []


if __name__ == "__main__":
    print("Available tiers:")
    print("1. dev (default)")
    print("2. devext")
    print("3. qa")
    print("4. prd")

    # ArcGIS Org
    USERNAME_DEV = "zeping_dev0"
    PASSWORD_DEV = "19990224lzpdsg!!!"

    USERNAME_DEVEXT = "zeping_dev0"
    PASSWORD_DEVEXT = "19990224lzpdsg!!!"

    USERNAME_QA = "zeping_dev0"
    PASSWORD_QA = "19990224lzpdsg!!!"

    USERNAME_PRD = "zeping_dev0"
    PASSWORD_PRD = "19990224lzpdsg!!!"

    # Endpoint library
    BASE_URL_DEV = "https://aiservices.dev.geocloud.com"
    BASE_URL_DEVEXT = "https://aiservicesdev-beta.arcgis.com"
    BASE_URL_QA = "https://aiservicesqa-beta.arcgis.com"
    BASE_URL_PRD = "https://aiservices-beta.arcgis.com"

    TOKEN_URL_DEV = "https://analysis-0.mapsdevext.arcgis.com/sharing/generateToken"
    TOKEN_URL_DEVEXT = "https://analysis-0.mapsdevext.arcgis.com/sharing/generateToken"
    TOKEN_URL_QA = "https://analysis-0.mapsqa.arcgis.com/sharing/generateToken"
    TOKEN_URL_PRD = "https://analysis-0.maps.arcgis.com/sharing/generateToken"

    os.environ["USERNAME_DEV"] = USERNAME_DEV
    os.environ["PASSWORD_DEV"] = PASSWORD_DEV
    os.environ["BASE_URL_DEV"] = BASE_URL_DEV
    os.environ["TOKEN_URL_DEV"] = TOKEN_URL_DEV

    os.environ["USERNAME_DEVEXT"] = USERNAME_DEVEXT
    os.environ["PASSWORD_DEVEXT"] = PASSWORD_DEVEXT
    os.environ["BASE_URL_DEVEXT"] = BASE_URL_DEVEXT
    os.environ["TOKEN_URL_DEVEXT"] = TOKEN_URL_DEVEXT

    os.environ["USERNAME_QA"] = USERNAME_QA
    os.environ["PASSWORD_QA"] = PASSWORD_QA
    os.environ["BASE_URL_QA"] = BASE_URL_QA
    os.environ["TOKEN_URL_QA"] = TOKEN_URL_QA

    os.environ["USERNAME_PRD"] = USERNAME_PRD
    os.environ["PASSWORD_PRD"] = PASSWORD_PRD
    os.environ["BASE_URL_PRD"] = BASE_URL_PRD
    os.environ["TOKEN_URL_PRD"] = TOKEN_URL_PRD

    tier_choice = "2"

    tier_map = {
        "1": "dev",
        "2": "devext",
        "3": "qa",
        "4": "prd",
        "": "dev"  # default
    }

    tier = tier_map.get(tier_choice, "dev")
    print(f"Using tier: {tier}")

    prompt_1_template = """
    Here is a spatial core concept:

    Core Concept: {core_name}
    Description: {core_desc}

    Task:
    Please list all Toolset names available in ArcGIS Pro Toolboxes that are potentially relevant to the above core concept.
    Just output the Toolset names—no need to list the individual tools yet.
    Please provide as many relevant Toolsets as possible, covering building, editing, analyzing, visualizing, or any spatial operations related to this concept.

    Response format:
    Toolset names:
    - Toolset Name 1
    - Toolset Name 2
    - Toolset Name 3
    ...
    """
    prompt_2_template = """
    You are provided with a list of ArcGIS Pro toolsets that may be relevant to the following spatial core concept:

    Core Concept:
    {name}: {desc}

    Toolset:
    {toolset}

    Task:
    For the toolset above, list all individual tools within that toolset that are relevant to the core concept.

    Only provide the tool names (no description or details needed at this step).

    Output the results in the following structured format, please do not include any additional text or explanations:
    <Tool Name 1>, <Tool Name 2>, <Tool Name 3>, ......
    """
    prompt_3_template="""
    Context:

    You are provided with the following ArcGIS Pro tool information:

    Toolset name: {toolset_name}

    Tool name: {tool_name}

    Task:

    For the specified tool, provide the following details in a structured format:

    1. Description:
    A concise summary (1–2 sentences) explaining what the tool does and its typical use cases.

    2. Parameters:
    List all input parameters. For each parameter, include:
        -Name
        -Description
        -Data type

    3. Derived Outputs:
    List all derived or output parameters with their name, description, and data type.

    4. Example ArcPy code:
    Provide a minimal working ArcPy script using the tool and its key parameters.
    The script should include all necessary imports and setup for execution.

    Response format:

    Toolset: <Toolset Name>

    Tool: <Tool Name>

    Description: <Concise description>

    Parameters:
    - <param1>: <explanation> Type: <Data Type>.
    - <param2>: <explanation> Type: <Data Type>.
    ...

    Derived Output:
    - <out_param1>: <explanation> Type: <Data Type>.
    - <out_param2>: <explanation> Type: <Data Type>.
    ...
    Example ArcPy code (include all the necessary imports and context to successfully run the code):
    ```python
    import arcpy
    arcpy.<toolbox_alias>.<tool_function>(
        <param1>=<value1>,
        <param2>=<value2>,
        ...
    )
    ...
    If the tool or parameters are not found, or there is no relevant information, respond only with:  `No information available.`
    """
    # prompt_3_template="""
    # Context:
    #
    # You are provided with the following ArcGIS Pro tool information:
    #
    # Toolset name: {toolset_name}
    #
    # Tool name: {tool_name}
    #
    # Task:
    #
    # For the specified tool, provide the following details in a structured format:
    #
    # 1. Description:
    # A concise summary (1–2 sentences) explaining what the tool does and its typical use cases.
    #
    # 2. Parameters:
    # List all input parameters. For each parameter, include:
    #     -Name
    #     -Description
    #     -Data type
    #
    # 3. Derived Outputs:
    # List all derived or output parameters with their name, description, and data type.
    #
    # 4. Example ArcPy code:
    # Provide a minimal working ArcPy script using the tool and its key parameters.
    # The script should include all necessary imports and setup for execution.
    #
    # Response format:
    #
    # Toolset: <Toolset Name>
    #
    # Tool: <Tool Name>
    #
    # Description: <Concise description>
    #
    # Parameters:
    # - <param1>: <explanation> Type: <Data Type>.
    # - <param2>: <explanation> Type: <Data Type>.
    # ...
    #
    # Derived Output:
    # - <out_param1>: <explanation> Type: <Data Type>.
    # - <out_param2>: <explanation> Type: <Data Type>.
    # ...
    # Example ArcPy code (include all the necessary imports and context to successfully run the code):
    # ```python
    # import arcpy
    # arcpy.<toolbox_alias>.<tool_function>(
    #     <param1>=<value1>,
    #     <param2>=<value2>,
    #     ...
    # )
    # ...
    # If the tool or parameters are not found, or there is no relevant information, respond only with:  `No information available.`
    # """
    # prompt_3_template = """
    #     Context:
    #
    #     You are provided with the following ArcGIS Pro tool information:
    #
    #     Toolset name: {toolset_name}
    #
    #     Tool name: {tool_name}
    #
    #     Spatial core concept: {core_name}: {core_desc}
    #
    #     Task:
    #
    #     For the specified tool, provide the following details in a structured format:
    #
    #     1. Description:
    #     A detailed summary explaining what the tool does and its typical use cases. Give some examples of how it can be used in spatial analysis or GIS workflows.
    #
    #     2. Parameters:
    #     List all input parameters. For each parameter, include:
    #         -Name
    #         -Description
    #         -Data type
    #
    #     3. Derived Outputs:
    #     List all derived or output parameters with their name, description, and data type.
    #
    #     Response format:
    #
    #     Toolset: <Toolset Name>
    #
    #     Tool: <Tool Name>
    #
    #     Description: <very detailed description with examples>
    #
    #     Parameters:
    #     - <param1>: <explanation> Type: <Data Type>.
    #     - <param2>: <explanation> Type: <Data Type>.
    #     ...
    #
    #     Derived Output:
    #     - <out_param1>: <explanation> Type: <Data Type>.
    #     - <out_param2>: <explanation> Type: <Data Type>.
    #
    #     If the tool or parameters are not found, or there is no relevant information, respond only with:  `No information available.`
    #     """
    try:
        token = get_token(tier)
        base_url, _ = get_urls_for_tier(tier)
        client = Client(base_url=base_url, timeout=60.0)
        skill_json = get_skill_ids(client, auth_token=token)
        for core in core_concepts:
            # prompt = input("> ")
            toolset_name_list = []
            context = {"kind": "DocAIAssistantRequest", "filters": {}}
            prompt = prompt_1_template.format(core_name=core["name"], core_desc=core["description"])
            for i in range(2):
                answer = chat_with_skill(
                    client=client, skill_id="doc_ai_assistant", message=prompt, auth_token=token, context=context
                )
                response_text = extract_response(answer, "doc_ai_assistant")
                toolset_names = extract_toolset_names(response_text)
                toolset_name_list.append(toolset_names)
            # Deduplicate toolset names across multiple responses
            toolset_names = list(set([item for sublist in toolset_name_list for item in sublist]))
            # toolset_names = llm_supervisor(str(toolset_names)) or toolset_names
            print(f"core concept {core} has {toolset_names}")

            for toolset in toolset_names:
                tool_name_list = []
                prompt_2 = prompt_2_template.format(toolset=toolset, name=core["name"], desc=core["description"])
                for i in range(2):
                    answer = chat_with_skill(
                        client=client, skill_id="doc_ai_assistant", message=prompt_2, auth_token=token, context=context
                    )
                    response_text = extract_response(answer, "doc_ai_assistant")
                    all_tools = extract_tools(response_text)
                    tool_name_list.append(all_tools)
                # Deduplicate tool names across multiple responses
                all_tools = list(set([item for sublist in tool_name_list for item in sublist]))
                # all_tools = llm_supervisor(str(all_tools)) or all_tools
                print(f"core concept {core} has {toolset_names}, which contains {all_tools}")
                for tool in all_tools:
                    print("processing tool:", tool)
                    prompt_3 = prompt_3_template.format(toolset_name=toolset, tool_name=tool.strip())
                    answer = chat_with_skill(
                        client=client, skill_id="doc_ai_assistant", message=prompt_3, auth_token=token, context=context
                    )
                    response_text = extract_response(answer, "doc_ai_assistant")
                    with open(f"{core['name']}_new.txt", "a") as f:
                        f.write(response_text + "\n")
                        print("file saved:", f"{core['name']}.txt")

    except Exception as e:
        print(f"Error: {e}")
        print("Make sure .env file is properly configured with the required endpoints and credentials.")
