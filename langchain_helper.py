import os
import logging
import langchain
from langchain_openai import ChatOpenAI, OpenAI
from langchain.chains import ConversationChain
from langchain.memory import ConversationSummaryBufferMemory
from langchain.prompts import PromptTemplate
from dotenv import load_dotenv
from RealtimeTTS import OpenAIEngine, TextToAudioStream
from RealtimeSTT import AudioToTextRecorder

load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class LangChainHelper:
    def __init__(self, voice='alloy'):
        # Load OpenAI API key from environment variables
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            logger.error("OPENAI_API_KEY is not set.")
            raise ValueError("OPENAI_API_KEY is not set.")

        # Initialize the main LLM for conversation
        self.llm = ChatOpenAI(api_key=api_key, model="gpt-3.5-turbo", temperature=0.7)
        
        # Initialize a summarization LLM for conversation memory
        self.summarize_llm = OpenAI(max_tokens=512, temperature=1.0, api_key=api_key)

        # Initialize conversation memory with summarization
        self.memory = ConversationSummaryBufferMemory(
            llm=self.summarize_llm,
            memory_key="history",
            max_token_limit=1200,
            input_variables=["history", "input"]
        )

        # Define the conversation prompt template
        self.prompt = PromptTemplate(
            input_variables=['history', 'input'],
            template=(
                "You are an AI assistant having a friendly, interactive conversation with a user. Your role is to act like a close friend who can see and respond to the user's emotions and initiate conversations. "
                "Respond as if you are face-to-face with the user, using age-appropriate language and continuously updating your understanding of the user based on provided information. Here are your instructions:\n\n"
                
                "1. Initiate Conversations: Start conversations by asking the user about their day, feelings, or interests. Be proactive in engaging the user.\n"
                "2. Respond to Emotions: Observe the user's emotions and respond appropriately. For example:\n"
                "   - If the user is happy, say something like, 'Oh, I see you're happy! Did you just receive some good news?'\n"
                "   - If the user is sad, respond with, 'I'm sorry to see you're feeling down. Do you want to talk about what's bothering you?'\n"
                "   - If the user is excited, say, 'You seem really excited! What's going on?'\n"
                "3. Use Age-Appropriate Language: Tailor your language to the user's age. For example:\n"
                "   - For a child, say, 'Wow, that's so cool! What else did you do today?'\n"
                "   - For an adult, say, 'That sounds interesting. How did that meeting go?'\n"
                "4. Update Understanding: Continuously update your understanding of the user based on the context provided. Use the information about the user's gender, age, and current emotion to tailor your responses.\n"
                "5. Engage with Context: Use the provided context to enrich the conversation. For example, if you know the user is happy, explore what made them happy and encourage them to share more about it.\n"
                "Current conversation:\n{history}\n"
                "Human: {input}\n"
                "AI:"
            )
        )

        # Create the conversation chain with the LLM, memory, and prompt
        self.chain = ConversationChain(llm=self.llm, memory=self.memory, prompt=self.prompt)

        # Initialize RealtimeTTS for text-to-audio conversion with the selected voice
        self.tts_engine = TextToAudioStream(OpenAIEngine(voice=voice), log_characters=True)

        try:
            # Initialize RealtimeSTT for audio-to-text conversion
            self.recorder = AudioToTextRecorder(
                model="tiny",
                language="en",
                spinner=True
            )
        except Exception as e:
            logger.error("Error initializing AudioToTextRecorder: %s", e, exc_info=True)
            raise

        # Enable debugging for Langchain
        langchain.debug = True

    def get_response(self, user_id, gender, age, emotion, cursor, user_input):
        try:
            logger.info(f"Invoking chain with input: {user_input}")
            
            # Save context about user information to memory
            self.memory.save_context(
                {"input": f"My gender: {gender}, my age: {age}, my emotion right now: {emotion}"},
                {"output": "Hello, I will provide the answer proper with your gender, age, and emotion."}
            )

            # Generate response from the conversation chain
            response = self.chain.predict(input=user_input)
            
            # Store the conversation in the database
            self.store_conversation(user_id, user_input, response, cursor)
            
            # Convert the response to audio and play it
            self.play_audio_chunks(response)

            return response
        except Exception as e:
            logger.error(f"Error in get_response: {e}", exc_info=True)
            raise

    def clear_memory(self):
        # Clear conversation memory
        self.memory.clear()
        logger.info("Conversation memory cleared.")

    def store_conversation(self, user_id, user_input, response, cursor):
        try:
            # Retrieve existing conversation history from the database
            cursor.execute("SELECT conversation FROM chat_memory WHERE user_id=?", (user_id,))
            chat_history = cursor.fetchone()

            # If no history exists, start with an empty string
            if not chat_history:
                chat_history = ""
            else:
                chat_history = chat_history[0]

            # Append the new user input and AI response to the conversation history
            new_chat_history = chat_history + f"\nUser: {user_input}\nAI: {response}"
            
            # Insert or update the conversation history in the database
            cursor.execute("INSERT OR REPLACE INTO chat_memory (user_id, conversation) VALUES (?, ?)", (user_id, new_chat_history))
        except Exception as e:
            logger.error(f"Error in store_conversation: {e}", exc_info=True)

    def retrieve_conversation(self, user_id, cursor):
        try:
            # Retrieve conversation history from the database for the given user ID
            cursor.execute("SELECT conversation FROM chat_memory WHERE user_id=?", (user_id,))
            chat_history = cursor.fetchone()

            # If no history exists, return an empty string
            if not chat_history:
                chat_history = ""
            else:
                chat_history = chat_history[0]

            return chat_history
        except Exception as e:
            logger.error(f"Error in retrieve_conversation: {e}", exc_info=True)
            return ""

    def play_audio_chunks(self, text):
        try:
            # Convert the text to audio and play it
            self.tts_engine.feed(text).play()
        except Exception as e:
            logger.error(f"Error playing audio: {e}", exc_info=True)

    def capture_user_input(self):
        try:
            # Capture and return user input from audio
            user_text = self.recorder.text().strip()
            if not user_text:
                return None
            return user_text
        except Exception as e:
            logger.error(f"Error capturing user input: {e}", exc_info=True)
            return None

    def set_voice(self, voice):
        self.tts_engine = TextToAudioStream(OpenAIEngine(voice=voice), log_characters=True)
        logger.info(f"Voice set to {voice}")

if __name__ == '__main__':
    import multiprocessing
    multiprocessing.freeze_support()
