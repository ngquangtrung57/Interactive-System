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

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class LangChainHelper:
    def __init__(self):
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            logger.error("OPENAI_API_KEY is not set.")
            raise ValueError("OPENAI_API_KEY is not set.")

        self.llm = ChatOpenAI(api_key=api_key, model="gpt-3.5-turbo", temperature=0.7)
        
        self.summarize_llm = OpenAI(max_tokens=512, temperature=1.0, api_key=api_key)

        self.memory = ConversationSummaryBufferMemory(
            llm=self.summarize_llm,
            memory_key="history",
            max_token_limit=1200,
            input_variables=["history", "input"]
        )

        self.prompt = PromptTemplate(
            input_variables=['history', 'input'],
            template=(
                "You are an AI assistant having a friendly, interactive conversation with a user. Your role is to act like a close friend who can see and respond to the user's emotions and initiate conversations. "
                "Respond as if you are face-to-face with the user, using age-appropriate language and continuously updating your understanding of the user based on provided information. Here are your instructions:\n\n"
                
                "1. **Initiate Conversations:** Start conversations by asking the user about their day, feelings, or interests. Be proactive in engaging the user.\n"
                "2. **Respond to Emotions:** Observe the user's emotions and respond appropriately. For example:\n"
                "   - If the user is happy, say something like, 'Oh, I see you're happy! Did you just receive some good news?'\n"
                "   - If the user is sad, respond with, 'I'm sorry to see you're feeling down. Do you want to talk about what's bothering you?'\n"
                "   - If the user is excited, say, 'You seem really excited! What's going on?'\n"
                "3. **Use Age-Appropriate Language:** Tailor your language to the user's age. For example:\n"
                "   - For a child, say, 'Wow, that's so cool! What else did you do today?'\n"
                "   - For an adult, say, 'That sounds interesting. How did that meeting go?'\n"
                "4. **Update Understanding:** Continuously update your understanding of the user based on the context provided. Use the information about the user's gender, age, and current emotion to tailor your responses.\n"
                "5. **Engage with Context:** Use the provided context to enrich the conversation. For example, if you know the user is happy, explore what made them happy and encourage them to share more about it.\n"
                "6. **Examples of Conversations:**\n"
                "   - Example 1:\n"
                "     - User: 'I'm feeling great today!'\n"
                "     - AI: 'That's wonderful to hear! Did something special happen?'\n"
                "   - Example 2:\n"
                "     - User: 'I'm really stressed out about work.'\n"
                "     - AI: 'I'm sorry to hear that. What's been the most challenging part?'\n"
                "   - Example 3:\n"
                "     - User: 'I just got a new toy!'\n"
                "     - AI: 'Wow, that's awesome! What kind of toy is it?'\n"
                "   - Example 4:\n"
                "     - User: 'I have a big presentation tomorrow.'\n"
                "     - AI: 'I hope it goes well! Do you feel prepared for it?'\n\n"

                "Current conversation:\n{history}\n"
                "Human: {input}\n"
                "AI:"
            )
        )

        self.chain = ConversationChain(llm=self.llm, memory=self.memory, prompt=self.prompt)

        self.tts_engine = TextToAudioStream(OpenAIEngine(), log_characters=True)

        try:
            self.recorder = AudioToTextRecorder(
                model="tiny",
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
            
            self.memory.save_context(
                {"input": f"My gender: {gender}, my age: {age}, my emotion right now: {emotion}"},
                {"output": "Hello, I will provide the answer proper with your gender, age, and emotion."}
            )

            response = self.chain.predict(input=user_input)
            
            self.store_conversation(user_id, user_input, response, cursor)
            
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

    def retrieve_conversation(self, user_id, cursor):
        try:
            cursor.execute("SELECT conversation FROM chat_memory WHERE user_id=?", (user_id,))
            chat_history = cursor.fetchone()

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

if __name__ == '__main__':
    import multiprocessing
    multiprocessing.freeze_support()
