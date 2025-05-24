Duolingo v2

Duolingo meets AI voice chats. Practice speaking a new language by actually talking to an AI—and getting real-time feedback with translations.

💡 Inspiration

We wanted to create a more immersive and engaging way to learn a new language. Apps like Duolingo are great, but they're often limited to tapping words or filling in blanks. Real-world fluency comes from actual conversation. So we built TalkLingua, where you can speak to an AI in a foreign language, get instant feedback, and see the English translation in real time. It's like having a personal tutor who's always available—and never judges.

🛠️ What it does

Duolingo-v2 lets you:

🗣️ Speak directly to an AI in Spanish (or any target language).
🔁 Get real-time responses from the AI in the same language.
📝 Receive a live transcript with English translation for every sentence.
You start a conversation, talk into your mic, and the AI listens, understands, and responds in kind—while translating everything so you know exactly what's going on.

⚙️ How we built it

We combined powerful APIs and services to create a seamless voice-first experience:

🎙 Daily: For real-time voice conversations over WebRTC.
🤖 Anthropic Claude 3 (via API): For generating natural Spanish responses and translating user input.
🧠 OpenAI (optional): For prompt tuning and future expansion into voice analysis.
🦾 Tavus: For personalized video generation, turning language exercises into visual conversation companions.
⚡ Bolt: For faster iteration of building frontend. 
🧪 Flask + SSE: Real-time Server-Sent Events streaming the AI and user utterances to the frontend.
We also used queueing and threading to handle continuous voice input/output with zero delay, and the anthropic.messages.create tool call system to ensure structured, tutor-style replies with both Spanish and English.

🧠 AI Prompting Highlights

We prompted Claude with:

"Act as a Spanish tutor. Respond in Spanish, then provide an English translation. Use structured tool calls only—no extra reasoning or chit-chat."
This made Claude return clear, concise, structured responses like:

{
  "spanish_response": "¿Qué te gustaría comer?",
  "english_translation": "What would you like to eat?"
}

🖥️ Live Demo

[YouTube Link](https://youtu.be/BfzSXjZbcsA)

🌍 Use Cases

Language learners who want to speak more than they tap.
Travelers prepping for real-life conversations.
ESL/ELL programs replacing passive lessons with active dialogue.
Anyone who wants an always-on speaking partner.

🚧 Challenges we ran into

Synchronizing audio input with real-time transcript delivery
Making Anthropic’s tool calling consistent and deterministic
Translating messy real-world speech into clean text
Managing real-time voice streams with low latency

🎉 Accomplishments we're proud of

Fully working AI-driven language partner
Structured translation pipeline (Spanish ↔ English)
Real-time voice chat with AI
Seamless integration between Anthropic, Daily, Tavus, and Bolt

🔮 What’s next

🧑‍🏫 Support for grammar correction and feedback
🌐 Multilingual support (French, German, Mandarin, etc.)
🧠 LLM fine-tuning on language pedagogy data
📱 Launch as a mobile-first experience

👥 Team

Karthik
Subra
