# newAPI.py

import logging
from time import sleep
from typing import Any, Dict, List, Optional
from datetime import datetime, timedelta
import json
import os
import dotenv

from httpx import Client, Response

logger = logging.getLogger(__name__)

dotenv.load_dotenv()


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


# def simple_answer(prompt: str, skill_id: str = "doc_chat", full_response: bool = False, context: Any = None):
#     token = get_token()
#     # Use a longer timeout for this client (60 seconds)
#     client = Client(base_url=BASE_URL, timeout=60.0)
#     answer = chat_with_skill(
#         client=client, skill_id=skill_id, message=prompt, auth_token=token, context=context
#     )
#     if full_response:
#         return answer
#     reply = None
#     try:
#         for interaction in answer:
#             if interaction["message"] is not None:
#                 reply = interaction["message"]
#         if reply == None:
#             reply = "Error accessing: No response"
#         return reply
#     except:
#         return None


if __name__ == "__main__":
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
    print("Available tiers:")
    print("1. dev (default)")
    print("2. devext")
    print("3. qa")
    print("4. prd")

    tier_choice = input("Select tier (1-4, or press Enter for default): ").strip()

    tier_map = {
        "1": "dev",
        "2": "devext",
        "3": "qa",
        "4": "prd",
        "": "dev"  # default
    }

    tier = tier_map.get(tier_choice, "dev")
    print(f"Using tier: {tier}")

    try:
        token = get_token(tier)
        base_url, _ = get_urls_for_tier(tier)
        client = Client(base_url=base_url, timeout=60.0)
        skill_json = get_skill_ids(client, auth_token=token)
        # print(skill_json)
        while True:
            prompt = input("> ")
            if not prompt.strip():
                prompt ='''Can you provide the Python parameters (hyperparameters) required for the 'Add Field' tool, so I can use it correctly in an arcpy script?  '''
            context = {"kind": "DocAIAssistantRequest", "filters": {}}
            answer = chat_with_skill(
                client=client, skill_id="doc_ai_assistant", message=prompt, auth_token=token, context=context
            )
            response_text = extract_response(answer, "doc_ai_assistant")
            print(response_text)
    except Exception as e:
        print(f"Error: {e}")
        print("Make sure .env file is properly configured with the required endpoints and credentials.")
