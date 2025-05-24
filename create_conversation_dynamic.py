import requests
import json
import sys
from flask import Flask, request, jsonify
from flask_cors import CORS
import time
import os

app = Flask(__name__)
CORS(app)

def create_conversation(language, topic, output_file):
    url = "https://tavusapi.com/v2/conversations"
    
    # Map languages to greetings
    greetings = {
        "english": "Hello! Let's talk about",
        "spanish": "¡Hola! Hablemos de",
        "french": "Bonjour! Parlons de"
    }

    # Map topics to topic phrases
    topic_phrases = {
        "restaurant": {
            "english": "ordering food at a restaurant",
            "spanish": "pedir comida en un restaurante",
            "french": "commander de la nourriture dans un restaurant"
        },
        "travel": {
            "english": "booking a hotel room",
            "spanish": "reservar una habitación de hotel",
            "french": "réserver une chambre d'hôtel"
        },
        "shopping": {
            "english": "shopping for clothes",
            "spanish": "comprar ropa",
            "french": "faire du shopping pour des vêtements"
        }
    }

    # Get appropriate greeting and topic phrase
    base_greeting = greetings.get(language.lower(), "Hello")
    topic_phrase = topic_phrases.get(topic, {}).get(language.lower(), topic_phrases["restaurant"][language.lower()])
    
    # Combine greeting and topic
    full_greeting = f"{base_greeting} {topic_phrase}"

    payload = {
        "replica_id": "rb17cf590e15",
        "conversation_name": f"Language Learning - {language.capitalize()} - {topic.capitalize()}",
        "conversational_context": f"You are a {language} language tutor helping someone learn about {topic_phrase}. Keep your responses short, friendly, and focused on the topic. Do not use more than 2 sentences.",
        "custom_greeting": full_greeting,
        "properties": {
            "language": language.lower()
        }
    }
    print(f"DEBUG: Payload: {payload}")
    headers = {
        "x-api-key": "94a3b175900048ffbd42d903a83c42ca",
        "Content-Type": "application/json"
    }

    try:
        response = requests.request("POST", url, json=payload, headers=headers)
        response_data = response.json()
        
        # Create output directory if it doesn't exist
        os.makedirs('conversation_responses', exist_ok=True)
        
        # Write response to file
        output_path = os.path.join('conversation_responses', output_file)
        with open(output_path, 'w') as f:
            json.dump(response_data, f, indent=2)
            
        return response_data
        
    except Exception as e:
        print(f"Error creating conversation: {e}")
        return None

@app.route('/create-conversation', methods=['POST'])
def create_conversation_endpoint():
    try:
        data = request.get_json()
        language = data.get('language', 'english')
        topic = data.get('topic', 'restaurant')
        print(f"Creating conversation for {language} and {topic}")
        # Create a unique filename for this conversation's response
        timestamp = int(time.time())
        output_file = f'conversation_{language}_{topic}_{timestamp}.json'
        
        # Create the conversation
        response_data = create_conversation(language, topic, output_file)
        
        if response_data and 'conversation_url' in response_data:
            return jsonify(response_data)
        else:
            return jsonify({'error': 'Failed to create conversation or get URL'}), 500
            
    except Exception as e:
        print(f"Error in create_conversation endpoint: {e}")
        return jsonify({'error': str(e)}), 500

if __name__ == "__main__":
    if len(sys.argv) > 1:
        # Command line usage
        if len(sys.argv) != 4:
            print("Usage: python create_conversation_dynamic.py <language> <topic> <output_file>")
            sys.exit(1)
            
        language = sys.argv[1]
        topic = sys.argv[2]
        output_file = sys.argv[3]
        create_conversation(language, topic, output_file)
    else:
        # Run as web server
        print("Starting conversation creation server on port 5001...")
        app.run(host='0.0.0.0', port=5002)