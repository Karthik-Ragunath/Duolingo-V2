import requests

url = "https://tavusapi.com/v2/conversations"

payload = {
    "replica_id": "r79e1c033f",
    "persona_id": "p5317866",
    "callback_url": "https://yourwebsite.com/webhook",
    "conversation_name": "A Meeting with Hassaan",
    "conversational_context": "You are about to talk to Hassaan, one of the cofounders of Tavus. He loves to talk about AI, startups, and racing cars.",
    "custom_greeting": "Hey there Hassaan, long time no see!",
    "properties": {
        "max_call_duration": 3600,
        "participant_left_timeout": 60,
        "participant_absent_timeout": 300,
        "enable_recording": True,
        "enable_closed_captions": True,
        "apply_greenscreen": True,
        "language": "english",
        "recording_s3_bucket_name": "conversation-recordings",
        "recording_s3_bucket_region": "us-east-1",
        "aws_assume_role_arn": ""
    }
}
headers = {
    "x-api-key": "<api-key>",
    "Content-Type": "application/json"
}

response = requests.request("POST", url, json=payload, headers=headers)

print(response.text)