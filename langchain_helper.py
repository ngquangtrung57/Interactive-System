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

# Load environment variables from a .env file
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class LangChainHelper:
    def __init__(self):
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
                "The following is a friendly conversation between a human and an AI. The AI is talkative and provides lots of specific details from its context. "
                "If the AI does not know the answer to a question, it truthfully says it does not know.\n\n"
                "You are AI, answer the user accordingly. "
                "Current conversation:\n{history}\n"
                "Human: {input}\n"
                "AI:"
            )
        )

        # Create the conversation chain with the LLM, memory, and prompt
        self.chain = ConversationChain(llm=self.llm, memory=self.memory, prompt=self.prompt)

        # Initialize RealtimeTTS for text-to-audio conversion
        self.tts_engine = TextToAudioStream(OpenAIEngine(), log_characters=True)

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
            self.memory.save_context({
                "input": f"My gender: {gender}, my age: {age}, my emotion right now: {emotion}",
                "output": "Hello, I will provide the answer proper with your gender, age, and emotion."
            })

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
