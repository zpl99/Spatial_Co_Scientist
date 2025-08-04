import logging
import os
import json
from datetime import datetime
from typing import Any, Dict, List, Optional
from httpx import Client, Response
from time import sleep

# Set environment variables directly for all tiers (dev, devext, qa, prd)
os.environ["USERNAME_DEV"] = "zeping_dev0"
os.environ["PASSWORD_DEV"] = "19990224lzpdsg!!!"
os.environ["BASE_URL_DEV"] = "https://aiservices.dev.geocloud.com"
os.environ["TOKEN_URL_DEV"] = "https://analysis-0.mapsdevext.arcgis.com/sharing/generateToken"

os.environ["USERNAME_DEVEXT"] = "zeping_dev0"
os.environ["PASSWORD_DEVEXT"] = "19990224lzpdsg!!!"
os.environ["BASE_URL_DEVEXT"] = "https://aiservicesdev-beta.arcgis.com"
os.environ["TOKEN_URL_DEVEXT"] = "https://analysis-0.mapsdevext.arcgis.com/sharing/generateToken"

os.environ["USERNAME_QA"] = "zeping_dev0"
os.environ["PASSWORD_QA"] = "19990224lzpdsg!!!"
os.environ["BASE_URL_QA"] = "https://aiservicesqa-beta.arcgis.com"
os.environ["TOKEN_URL_QA"] = "https://analysis-0.mapsqa.arcgis.com/sharing/generateToken"

os.environ["USERNAME_PRD"] = "zeping_dev0"
os.environ["PASSWORD_PRD"] = "19990224lzpdsg!!!"
os.environ["BASE_URL_PRD"] = "https://aiservices-beta.arcgis.com"
os.environ["TOKEN_URL_PRD"] = "https://analysis-0.maps.arcgis.com/sharing/generateToken"

logger = logging.getLogger(__name__)

def get_urls_for_tier(tier: str = "dev") -> tuple[str, str]:
    """
    Get base URL and token URL for the specified tier.
    """
    base_url_key = f"BASE_URL_{tier.upper()}"
    token_url_key = f"TOKEN_URL_{tier.upper()}"
    base_url = os.getenv(base_url_key)
    token_url = os.getenv(token_url_key)
    if not base_url or not token_url:
        raise ValueError(
            f"Missing environment variables for tier '{tier}'. Check {base_url_key} and {token_url_key} values.")
    return base_url, token_url

def get_fresh_token(tier: str = "dev") -> Dict[str, Any]:
    """
    Get a new token using credentials for the specified tier.
    """
    _, token_url = get_urls_for_tier(tier)
    username = os.getenv(f"USERNAME_{tier.upper()}")
    password = os.getenv(f"PASSWORD_{tier.upper()}")
    if not username or not password:
        raise ValueError(
            f"Missing credentials for tier '{tier}'. Check USERNAME_{tier.upper()} and PASSWORD_{tier.upper()}.")
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
    return response.json()

def get_token(tier: str = "dev") -> str:
    """
    Return a valid token for the specified tier, reading from local file cache or requesting a new one if needed.
    """
    token_file = f"auth_token_{tier}.json"
    if os.path.exists(token_file):
        with open(token_file, 'r') as f:
            token_data = json.load(f)
        expires = datetime.fromtimestamp(token_data['expires'] / 1000)
        if datetime.now() < expires:
            return token_data['token']
    token_data = get_fresh_token(tier)
    with open(token_file, 'w') as f:
        json.dump(token_data, f)
    return token_data['token']

def chat_with_skill(
    client: Client,
    skill_id: Optional[str],
    message: str,
    auth_token: str,
    context: Any = None,
) -> List[Dict]:
    """
    Send a message to the ArcGIS Doc AI assistant skill and return the full response history.
    """
    url = f"/skills/{skill_id}/chat" if skill_id else "/chat"
    chat_request = {
        "message": message,
        "context": context,
    }
    responses: List[Dict] = []
    resp: Response = client.post(
        url,
        json=chat_request,
        headers={"token": f"{auth_token}"},
    )
    if resp.status_code != 200:
        raise Exception(f"Failed to send message: {message}. Response: {resp.content}")
    chat_response = resp.json()
    responses.append(chat_response)
    has_next = chat_response.get("hasMore")
    conversation_id = chat_response.get("conversationId")
    inquiry_id = chat_response.get("inquiryId")
    ack_sequence_number = chat_response.get("sequenceNumber")
    sleep(.5)
    while has_next:
        poll_chat_request = {
            "conversationId": conversation_id,
            "inquiryId": inquiry_id,
            "ackSequenceNumber": ack_sequence_number,
        }
        resp = client.post(
            url,
            json=poll_chat_request,
            headers={"token": f"{auth_token}"},
        )
        if resp.status_code != 200:
            logger.error("Failed to poll chat.")
            return None
        chat_response = resp.json()
        ack_sequence_number = chat_response["sequenceNumber"]
        responses.append(chat_response)
        has_next = chat_response["hasMore"]
        sleep(1)
    return responses

def extract_response(responses: List[Dict], skill_id: str) -> str:
    """
    Extract the answer text from the full response.
    """
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

def ask_doc_ai(query: str, tier: str = "devext", skill_id: str = "doc_ai_assistant", context: Optional[dict] = None) -> str:

    """
    Main interface for asking a question to the ArcGIS Doc AI assistant.
    Input your question, get a text answer.
    """

    if context is None:
        context = {"kind": "DocAIAssistantRequest", "filters": {}}
    token = get_token(tier)
    base_url, _ = get_urls_for_tier(tier)
    client = Client(base_url=base_url, timeout=60.0)
    try:
        responses = chat_with_skill(
            client=client, skill_id=skill_id, message=query, auth_token=token, context=context
        )
        return extract_response(responses, skill_id)
    except Exception as e:
        logger.error(f"Error in ask_doc_ai: {e}")
        return f"Error accessing: {str(e)}"

# Example usage
if __name__ == "__main__":
    prompt = """
    This is the user task: {'task_inst': 'Analyze and visualize Elk movements in the given dataset. Estimate home ranges and assess habitat preferences using spatial analysis techniques. Identify the spatial clusters of Elk movements. Document the findings with maps and visualizations. Save the figure as "pred_results/Elk_Analysis.png".',
 'dataset_path': 'ElkMovement/',
 'dataset_folder_tree': '|-- ElkMovement/\n|---- Elk_in_Southwestern_Alberta_2009.geojson',
 'dataset_preview': '[START Preview of ElkMovement/Elk_in_Southwestern_Alberta_2009.geojson]\n{"type":"FeatureCollection","features":[{"type":"Feature","id":1,"geometry":{"type":"Point","coordinates":[-114.19111179959417,49.536741600111178]},"properties":{"OBJECTID":1,"timestamp":"2009-01-01 01:00:37","long":-114.1911118,"lat":49.536741599999999,"comments":"Carbondale","external_t":-5,"dop":2.3999999999999999,"fix_type_r":"3D","satellite_":0,"height":1375.1900000000001,"crc_status":" ","outlier_ma":0,"sensor_typ":"gps","individual":"Cervus elaphus","tag_ident":"856","ind_ident":"E001","study_name":"Elk in southwestern Alberta","date":1709164800000,"time":" ","timestamp_Converted":1230771637000,"summer_indicator":1}},{"type":"Feature","id":2,"geometry":{"type":"Point","coordinates":[-114.1916239994119,49.536505999952517]},"properties":{"OBJECTID":2,"timestamp":"2009-01-01 03:00:52","long":-114.191624,"lat":49.536506000000003,"comments":"Carbondale","external_t":-6,"dop":2.3999999999999999,"fix_type_r":"3D","satellite_":0,"height":1375.2,"crc_status":" ","outlier_ma":0,"sensor_typ":"gps","individual":"Cervus elaphus","tag_ident":"856","ind_ident":"E001","study_name":"Elk in southwestern Alberta","date":1709164800000,"time":" ","timestamp_Converted":1230778852000,"summer_indicator":1}},{"type":"Feature","id":3,"geometry":{"type":"Point","coordinates":[-114.19169140075056,49.536571800069581]},"properties":{"OBJECTID":3,"timestamp":"2009-01-01 05:00:49","long":-114.1916914,"lat":49.536571799999997,"comments":"Carbondale","external_t":-6,"dop":5.6000000000000014,"fix_type_r":"3D","satellite_":0,"height":1382.0999999999999,"crc_status":" ","outlier_ma":0,"sensor_typ":"gps","individual":"Cervus elaphus","tag_ident":"856","ind_ident":"E001","study_name":"Elk in southwestern Alberta","date":1709164800000,"time":" ","timestamp_Converted":1230786049000,"summer_indicator":1}},...]}\n[END Preview of ElkMovement/Elk_in_Southwestern_Alberta_2009.geojson]',
 'output_fname': 'pred_results/Elk_Analysis.png'}, this is one"""
    answer = ask_doc_ai(prompt)
    print(answer)
