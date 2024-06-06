import streamlit as st
import sqlite3
from langchain_helper import LangChainHelper

def main():
    # Initialize the LangChainHelper
    helper = LangChainHelper()

    # Connect to SQLite database
    conn = sqlite3.connect('chat_memory.db')
    cursor = conn.cursor()
    cursor.execute('''CREATE TABLE IF NOT EXISTS chat_memory (user_id TEXT PRIMARY KEY, conversation TEXT)''')
    conn.commit()

    # Streamlit app
    st.title("Interactive Talkbot")

    # User ID input
    user_id = st.text_input("Enter your user ID", "test_user")

    # Conversation display
    conversation_history = st.empty()

    if "conversation_started" not in st.session_state:
        st.session_state["conversation_started"] = False
        st.session_state["pause"] = False
        st.session_state["conversation_history"] = []

    # Start button
    if not st.session_state["conversation_started"] and st.button("Start"):
        st.session_state["conversation_started"] = True
        st.session_state["pause"] = False
        st.session_state["conversation_history"] = []
        helper.clear_memory()
        
        # Initial greeting
        greeting = "Hello, how can I help you today?"
        st.session_state["conversation_history"].append(f"AI: {greeting}")
        conversation_history.text("\n".join(st.session_state["conversation_history"]))
        helper.play_audio_chunks(greeting)

    # Pause and Resume buttons
    if st.session_state["conversation_started"]:
        if not st.session_state["pause"]:
            if st.button("Pause"):
                st.session_state["pause"] = True
                helper.play_audio_chunks("Listening paused")
                st.write("Listening paused.")
        else:
            if st.button("Resume"):
                st.session_state["pause"] = False
                helper.play_audio_chunks("Listening resumed")
                st.write("Listening resumed.")

        # End button
        if st.button("End"):
            st.session_state["conversation_started"] = False
            st.session_state["pause"] = False
            st.write("Conversation ended.")
            helper.clear_memory()
            conn.commit()

    # Capture and process user input
    while st.session_state["conversation_started"] and not st.session_state["pause"]:
        user_input = helper.capture_user_input()
        if user_input:
            st.session_state["conversation_history"].append(f"{user_id}: {user_input}")

            # Update conversation display with user input
            conversation_history.text("\n".join(st.session_state["conversation_history"]))

            # Indicate the bot is generating a response
            st.write("Getting response from LLM...")

            # Mock CV data
            gender = "male"
            age = 25
            emotion = "happy"

            # Generate response
            response = helper.get_response(user_id=user_id, gender=gender, age=age, emotion=emotion, cursor=cursor, user_input=user_input)
            st.session_state["conversation_history"].append(f"AI: {response}")

            # Update conversation display with AI response
            conversation_history.text("\n".join(st.session_state["conversation_history"]))
        else:
            st.write("Could not capture your input. Please try again.")

    # Close the database connection
    conn.close()

if __name__ == "__main__":
    import multiprocessing
    multiprocessing.freeze_support()
    main()
