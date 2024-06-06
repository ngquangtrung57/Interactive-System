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

from prompts import SUMMARY_PROMPT

load_dotenv()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class LangChainHelper:
    def __init__(self):
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            logger.error("OPENAI_API_KEY is not set.")
            raise ValueError("OPENAI_API_KEY is not set.")

        # Initialize LLMs
        self.llm = ChatOpenAI(api_key=api_key, model="gpt-3.5-turbo", temperature=0.7)
        self.summarize_llm = OpenAI(max_tokens=512, temperature=1.0, api_key=api_key)

        # Initialize conversation memory
        self.memory = ConversationSummaryBufferMemory(
            llm=self.summarize_llm,
            memory_key="history",
            max_token_limit=1200,
            prompt=SUMMARY_PROMPT,
            input_variables=["history", "input"]
        )

        # Define conversation prompt
        self.prompt = PromptTemplate(
            input_variables=['history', 'input'],
            template=(
                "The following is a friendly conversation between a human and an AI. The AI is talkative and provides lots of specific details from its context. "
                "If the AI does not know the answer to a question, it truthfully says it does not know.\n\n"
                "You are AI, answer the user accordingly. "
                "The input may include the type of verification with the information to answer (e.g., SUPPORTS, REFUTES, NOT ENOUGH INFO). Use only the verified claims and partially verified.\n\n"
                "Current conversation:\n{history}\n"
                "Human: {input}\n"
                "AI:"
            )
        )

        self.chain = ConversationChain(llm=self.llm, memory=self.memory, prompt=self.prompt)

        # Initialize RealtimeTTS TextToAudioStream
        self.tts_engine = TextToAudioStream(OpenAIEngine(), log_characters=True)

        try:
            self.recorder = AudioToTextRecorder(
                model="small",
                language="en",
                spinner=True
            )
        except Exception as e:
            logger.error("Error initializing AudioToTextRecorder: %s", e, exc_info=True)
            raise

        langchain.debug = True

    def get_response(self, user_id, gender, age, emotion, cursor, user_input):
        try:
            logger.info(f"Invoking chain with input: {user_input}")
            response = self.chain.predict(input=user_input)
            self.store_conversation(user_id, user_input, response, cursor)
            # Convert response to audio and play
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
            cursor.execute("SELECT conversation FROM chat_memory WHERE user_id=?", (user_id,))
            chat_history = cursor.fetchone()

            if not chat_history:
                chat_history = ""
            else:
                chat_history = chat_history[0]

            new_chat_history = chat_history + f"\nUser: {user_input}\nAI: {response}"
            cursor.execute("INSERT OR REPLACE INTO chat_memory (user_id, conversation) VALUES (?, ?)", (user_id, new_chat_history))
        except Exception as e:
            logger.error(f"Error in store_conversation: {e}", exc_info=True)
    
    def play_audio_chunks(self, text):
        try:
            # Play the text as audio
            self.tts_engine.feed(text).play()
        except Exception as e:
            logger.error(f"Error playing audio: {e}", exc_info=True)
            
    def capture_user_input(self):
        try:
            user_text = self.recorder.text().strip()
            if not user_text:
                return None
            return user_text
        except Exception as e:
            logger.error(f"Error capturing user input: {e}", exc_info=True)
            return None
