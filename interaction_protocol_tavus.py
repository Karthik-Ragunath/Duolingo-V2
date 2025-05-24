import argparse
import json
import os
import time
import wave
from enum import Enum
from typing import Any, Mapping, Optional
import requests
from dotenv import load_dotenv
from daily import CallClient, Daily, EventHandler, VirtualMicrophoneDevice
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, GenerationConfig
import re

# # model_name = "deepseek-ai/deepseek-math-7b-instruct"
# model_name = "/home/ubuntu/karthik-ragunath-ananda-kumar-utah/deepseek-checkpoints/deepseek-math-7b-rl"
# tokenizer = AutoTokenizer.from_pretrained(model_name)
# model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.bfloat16, device_map="auto")
# model.generation_config = GenerationConfig.from_pretrained(model_name)
# model.generation_config.pad_token_id = model.generation_config.eos_token_id

load_dotenv()

DAILY_API_KEY = os.getenv("DAILY_API_KEY", "")
TAVUS_API_KEY = os.getenv("TAVUS_API_KEY")
ENV_TO_TEST = os.getenv("ENV_TO_TEST", "prod")

assistant_utterance = None
gpu_joined = False
assistant_utterance_time = None
warm_boot_time = None
questions_seen = {}
# global_lock = False

class TestType(Enum):
    FULL = "full"
    # ECHO = "echo"


class RoomHandler(EventHandler):
    def __init__(self):
        super().__init__()

    def on_app_message(self, message, sender: str) -> None:
        global client_global, conversation_id_global, conversation_url_global
        # print(f"Incoming app message from {sender}: {message}")
        try:
            if isinstance(message, str):
                json_message = json.loads(message)
            else:
                json_message = message
        except Exception as e:
            print(f"Error parsing message: {e}")
        # if json_message["event_type"] == "conversation.utterance":
        #     print(f"Utterance: {json_message['properties']['speech']}")
        # if json_message["event_type"] == "conversation.perception_tool_call":
        #     print(f"Perception tool call: {json_message.keys()}")
        #     print("Property Keys: ", json_message["properties"].keys())
        #     print("Arguments: ", json_message["properties"]["arguments"])
        #     print("Name: ", json_message["properties"]["name"])
        #     if json_message["properties"]["name"] == "notify_if_math_problem_found":
        #         question = json_message["properties"]["arguments"]["question"]
        #         parsed_question = question.split("=")[0].strip()
        #         if parsed_question not in questions_seen:
        #             questions_seen[parsed_question] = True
        #             send_text_echo(
        #                 client_global,
        #                 conversation_id_global,
        #                 "oh interesting problem, let me think about it for a moment",
        #             )
        #             time.sleep(3)
        #             result = call_deepseek_llm(parsed_question)
        #             # 5-10s of thinking
        #             send_text_echo(
        #                 client_global,
        #                 conversation_id_global,
        #                 f"i think i got it, here's how i got to the answer: {result}",
        #             )
        #     elif json_message["properties"]["name"] == "notify_if_cat_seen":
        #         send_text_echo(
        #             client_global, conversation_id_global, "oh no a cat! i'm scared"
        #         )
        #     elif json_message["properties"]["name"] == "notify_if_dog_seen":
        #         send_text_echo(
        #             client_global, conversation_id_global, "oh no a dog! i'm scared"
        #         )
        if (
            json_message["event_type"] == "conversation.utterance"
            and json_message["properties"]["role"] == "replica"
        ):
            global assistant_utterance, assistant_utterance_time
            assistant_utterance = json_message["properties"]["speech"]
            assistant_utterance_time = time.time()
            print(f"Assistant utterance: {assistant_utterance}")
        elif json_message["event_type"] == "system.replica_joined":
            global gpu_joined, warm_boot_time
            gpu_joined = True
            warm_boot_time = time.time()

def clean_math_text(text):
    """
    Remove LaTeX formatting and special symbols from mathematical text
    
    Args:
        text (str): Text containing LaTeX math symbols
        
    Returns:
        str: Cleaned text with special symbols replaced or removed
    """
    # Remove dollar sign delimiters (LaTeX math mode)
    text = re.sub(r'\$', '', text)
    
    # Replace LaTeX symbols with plain text equivalents
    replacements = {
        r'\\div': '/',           # Division symbol
        r'\\cdot': '*',          # Multiplication dot
        r'\\times': '*',         # Multiplication x
        r'\\frac{([^}]+)}{([^}]+)}': r'\1/\2',  # Fractions like \frac{a}{b} to a/b
        r'\\sqrt{([^}]+)}': r'sqrt(\1)',        # Square root
        r'\\sqrt\[([^]]+)\]{([^}]+)}': r'\1-root(\2)',  # nth root
        r'\\left\(': '(',        # Left parenthesis
        r'\\right\)': ')',       # Right parenthesis
        r'\\left\[': '[',        # Left bracket
        r'\\right\]': ']',       # Right bracket
        r'\\infty': 'infinity',  # Infinity symbol
        r'\\pi': 'pi',           # Pi symbol
        r'\\approx': '≈',        # Approximately equal
        r'\\neq': '≠',           # Not equal
        r'\\leq': '≤',           # Less than or equal
        r'\\geq': '≥',           # Greater than or equal
        r'\\boxed{([^}]+)}': r'\1'  # Remove boxed content formatting
    }
    
    for pattern, replacement in replacements.items():
        text = re.sub(pattern, replacement, text)
    
    # Process superscripts (^) for powers
    text = re.sub(r'(\w+)\^(\w+)', r'\1^\2', text)  # Preserve powers for clarity
    
    # Remove any remaining LaTeX commands
    text = re.sub(r'\\[a-zA-Z]+', '', text)
    
    return text

# def call_deepseek_llm(question):
#     parse_question = question.split("=")[0].strip()
#     messages = [
#         {"role": "user", "content": f"what is {parse_question}, solve this problem and put your final answer within " + "\\boxed{}"}
#     ]
#     input_tensor = tokenizer.apply_chat_template(messages, add_generation_prompt=True, return_tensors="pt")
#     outputs = model.generate(input_tensor.to(model.device), max_new_tokens=100)

#     result = tokenizer.decode(outputs[0][input_tensor.shape[1]:], skip_special_tokens=True)
#     cleaned_result = clean_math_text(result)
#     print(cleaned_result)
#     return cleaned_result


def call_joined(join_data: Optional[Mapping[str, Any]], client_error: Optional[str]):
    if client_error:
        raise RuntimeError(f"Bot failed to join call: {client_error}, {join_data}")
    else:
        print(f"call_joined ran successfully")


def join_room(call_client: CallClient, url: str, conversation_id: str):
    try:
        call_client.join(
            meeting_url=url,
            meeting_token=get_meeting_token(
                conversation_id, DAILY_API_KEY, True, None, False
            ),
            client_settings={
                "inputs": {
                    "microphone": {
                        "isEnabled": True,
                        "settings": {"deviceId": "microphone"},
                    }
                },
                "publishing": {
                    "microphone": {
                        "isPublishing": True,
                        "sendSettings": {"channelConfig": "mono", "bitrate": 16000},
                    }
                },
            },
            completion=call_joined,
        )
        print(f"Joined room: {url}")
    except Exception as e:
        print(f"Error joining room: {e}")
        raise


def play_audio(virtual_mic: VirtualMicrophoneDevice, audio_path: str):
    try:
        # Verify file exists and is readable
        if not os.path.exists(audio_path):
            raise FileNotFoundError(f"Audio file not found: {audio_path}")

        with wave.open(audio_path, "rb") as wave_file:
            # Verify audio format
            if wave_file.getsampwidth() != 2:  # 16-bit = 2 bytes
                raise ValueError("Audio must be 16-bit PCM")
            if wave_file.getnchannels() != virtual_mic.channels:
                raise ValueError(f"Audio must have {virtual_mic.channels} channels")

            # Read the actual frames
            audio_frames = wave_file.readframes(wave_file.getnframes())

            # Write frames and wait for completion
            frames_written = virtual_mic.write_frames(
                audio_frames,
                None,
                # lambda x: print(
                #     f"WROTE FRAMES: {x} (Expected: {len(audio_frames) // 2})"
                # ),
            )

            # Since we're using non-blocking mode, add a small delay
            time.sleep(0.1)

            if frames_written == 0:
                raise ValueError(f"No frames were written to virtual microphone")

            print(f"Played audio: {audio_path} ({frames_written} frames)")
    except Exception as e:
        print(f"Error playing audio: {e}")
        raise


def send_text_echo(
    call_client: CallClient,
    conversation_id: str,
    text: str = "This is the text the replica will speak.",
):
    call_client.send_app_message(
        {
            "message_type": "conversation",
            "event_type": "conversation.echo",
            "conversation_id": conversation_id,
            "properties": {"text": text},
        }
    )


def send_audio_echo(
    call_client: CallClient, conversation_id: str, audio_path: str = "test_audio.wav"
):
    try:
        with open(audio_path, "rb") as audio_file:
            audio_bytes = audio_file.read()
            call_client.send_app_message(
                {
                    "message_type": "conversation",
                    "event_type": "conversation.echo",
                    "conversation_id": conversation_id,
                    "properties": {
                        "modality": "audio",
                        "audio": audio_bytes,
                        "done": True,
                    },
                }
            )
            print(f"Played audio: {audio_path}")
    except Exception as e:
        print(f"Error playing audio: {e}")
        raise


def get_meeting_token(
    room_name: str,
    daily_api_key: str,
    has_presence: bool,
    token_expiry: Optional[float],
    enable_transcription: bool,
) -> str:
    assert daily_api_key, "Must provide `DAILY_API_KEY` env var"

    if not token_expiry:
        token_expiry = time.time() + 600
    res = requests.post(
        "https://api.daily.co/v1/meeting-tokens",
        headers={"Authorization": f"Bearer {daily_api_key}"},
        json={
            "properties": {
                "room_name": room_name,
                "is_owner": True,
                "enable_live_captions_ui": enable_transcription,
                "exp": token_expiry,
                "permissions": {"hasPresence": has_presence},
            }
        },
    )
    assert res.status_code == 200, f"Unable to create meeting token: {res.text}"

    meeting_token = res.json()["token"]
    return str(meeting_token)


def init_daily():
    Daily.init()
    output_handler = RoomHandler()
    client = CallClient(event_handler=output_handler)
    vmic = Daily.create_microphone_device("microphone", non_blocking=True, channels=1)
    return client, vmic


def init_daily_client():
    Daily.init()
    output_handler = RoomHandler()
    client = CallClient(event_handler=output_handler)
    return client


def run_heartbeat(conversation_url: str, conversation_id: str):
    global client_global, conversation_id_global, conversation_url_global
    client_global = init_daily_client()
    # Join the room
    conversation_id_global = conversation_id
    conversation_url_global = conversation_url
    join_room(client_global, conversation_url_global, conversation_id_global)
    while True:
        time.sleep(1)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--conversation_id", type=str, required=True, help="Conversation ID"
    )
    parser.add_argument(
        "--conversation_url", type=str, required=True, help="Conversation URL"
    )
    args = parser.parse_args()
    run_heartbeat(args.conversation_url, args.conversation_id)


if __name__ == "__main__":
    main()
