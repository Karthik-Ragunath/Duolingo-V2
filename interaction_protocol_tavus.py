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
from flask import Flask, Response, stream_with_context
from flask_cors import CORS
from queue import Queue
from threading import Thread
from anthropic import Anthropic

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
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY", "")  # Add your Anthropic API key to .env file

# Initialize Anthropic client
anthropic = Anthropic(api_key=ANTHROPIC_API_KEY)

assistant_utterance = None
gpu_joined = False
assistant_utterance_time = None
warm_boot_time = None
questions_seen = {}
# global_lock = False

# --- Added for SSE ---
utterance_queue = Queue()
app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

def get_claude_response(utterance: str) -> str:
    """Get a response from Claude for the given utterance."""
    topic = "Ordering food at a restaurant"
    client = anthropic.Anthropic()

    try:
        message = client.messages.create(
            model="claude-3-opus-20240229",
            max_tokens=1024,
            temperature=0.7,
            system=f"Act as a patient and encouraging Spanish language tutor. Engage in a conversation with me, correcting my grammar and vocabulary as needed. " \
                   f"Respond to my messages in Spanish, and provide English translations or explanations when necessary to help me understand. Our conversation topic is {topic}.",
            messages=[
                {"role": "user", "content": utterance}
            ],
            tools=[
                {
                    "type": "function",
                    "name": "generate_response",
                    "description": "Generate a response in Spanish with an English translation",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "spanish_response": {
                                "type": "string",
                                "description": "The response in Spanish"
                            },
                            "english_translation": {
                                "type": "string",
                                "description": "The English translation of the response"
                            }
                        },
                        "required": ["spanish_response", "english_translation"]
                    }
                }
            ]
        )
        # Check if Claude used the tool
        if message.content[0].type == "tool_use":
            tool_use = message.content[0]
            if tool_use.name == "generate_response":
                # Extract the Spanish response and English translation
                spanish_response = tool_use.parameters["spanish_response"]
                english_translation = tool_use.parameters["english_translation"]
                return f"Spanish: {spanish_response}, English: {english_translation}"
        else:
            # Handle the case where Claude didn't use the tool
            return message.content[0].text

    except Exception as e:
        print(f"Error getting Claude response: {e}")
        return f"I apologize, but I encountered an error processing your message: {str(e)}"

@app.route('/listen-utterances')
def listen_utterances():
    def event_stream():
        while True:
            try:
                # Wait for a new utterance from the queue
                utterance = utterance_queue.get()
                print(f"DEBUG: Got utterance from queue: {utterance}")
                
                if utterance is None:  # Allow graceful shutdown if needed
                    print("DEBUG: Received None utterance, breaking stream")
                    break
                
                # First, send the user's utterance
                user_message = {
                    "type": "user_utterance",
                    "text": utterance
                }
                yield f"data: {json.dumps(user_message)}\n\n"
                print("DEBUG: User message yielded successfully")

                # Get Claude's response
                claude_response = get_claude_response(utterance)
                ai_message = {
                    "type": "ai_response",
                    "text": claude_response
                }
                yield f"data: {json.dumps(ai_message)}\n\n"
                print("DEBUG: AI response yielded successfully")
                
                utterance_queue.task_done()
            except Exception as e:
                print(f"ERROR in event_stream: {str(e)}")
                continue
    
    return Response(
        stream_with_context(event_stream()),
        mimetype='text/event-stream',
        headers={
            'Cache-Control': 'no-cache',
            'Connection': 'keep-alive',
            'X-Accel-Buffering': 'no'
        }
    )

class TestType(Enum):
    FULL = "full"
    # ECHO = "echo"


class RoomHandler(EventHandler):
    def __init__(self):
        super().__init__()

    def on_app_message(self, message, sender: str) -> None:
        global client_global, conversation_id_global, conversation_url_global
        try:
            if isinstance(message, str):
                json_message = json.loads(message)
            else:
                json_message = message
        except Exception as e:
            print(f"Error parsing message: {e}")
            return

        if json_message["event_type"] == "conversation.utterance":
            utterance_text = json_message['properties']['speech']
            role_text = json_message["properties"]["role"]
            print(f"DEBUG: Received utterance - Text: {utterance_text}, Role: {role_text}")
            
            if role_text == "replica":  # Changed to capture non-replica speech
                print(f"DEBUG: Queueing utterance from {role_text}")
                utterance_queue.put(utterance_text)
                print(f"DEBUG: Successfully queued utterance")
            else:
                print(f"DEBUG: Skipping replica utterance")
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

    # --- Added for SSE ---
    # Start Flask server in a background thread
    # Use port 5001 as configured in index.html
    flask_thread = Thread(target=lambda: app.run(host='0.0.0.0', port=5001, debug=False, use_reloader=False), daemon=True)
    flask_thread.start()
    print("Flask SSE server started on port 5001")
    # --- End SSE Additions ---

    run_heartbeat(args.conversation_url, args.conversation_id)


if __name__ == "__main__":
    main()
