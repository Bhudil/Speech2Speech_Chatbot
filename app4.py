import streamlit as st
from groq import Groq
from elevenlabs.client import ElevenLabs
from elevenlabs import play
from streamlit_mic_recorder import speech_to_text

st.set_page_config(page_title="🎙️ Voice Bot", layout="wide")
st.title("🎙️ Speech Bot")
st.sidebar.title("Speak with LLMs")

def synthesize_speech(text):
    """
    Synthesize speech using ElevenLabs API
    
    Args:
        text (str): Text to be converted to speech
    
    Returns:
        Audio playback of the input text
    """
    client = ElevenLabs(api_key=ELEVENLABS_API_KEY)    
    audio = client.text_to_speech.convert(
        voice_id="IKne3meq5aSn9XLyUdCD",
        output_format="mp3_44100_128",
        text=text,
        model_id="eleven_multilingual_v2"	
    )
    return play(audio)

def llm_selector():
    """
    Select Groq LLM model
    
    Returns:
        str: Selected model name
    """
    groq_models = ['llama3-8b-8192', 'mixtral-8x7b-32768', 'gemma-7b-it']
    
    with st.sidebar:
        return st.selectbox("LLM", groq_models)

def generate_groq_response(text, model):
    """
    Generate response using Groq API
    
    Args:
        text (str): User input
        model (str): Groq model to use
    
    Returns:
        str: AI-generated response
    """
    client = Groq(api_key=GROQ_API_KEY)
    
    chat_completion = client.chat.completions.create(
        messages=[
            {
                "role": "system",
                "content": "You are a helpful assistant. Always provide short, concise and accurate answers."
            },
            {
                "role": "user",
                "content": text,
            }
        ],
        model=model,
    )
    return chat_completion.choices[0].message.content

def print_txt(text):
    """
    Print text using Streamlit markdown
    
    Args:
        text (str): Text to be printed
    """
    st.markdown(text, unsafe_allow_html=True)

def print_chat_message(message):
    """
    Print chat message with appropriate avatar
    
    Args:
        message (dict): Message containing role and content
    """
    text = message["content"]
    if message["role"] == "user":
        with st.chat_message("user", avatar="🎙️"):
            print_txt(text)
    else:
        with st.chat_message("assistant", avatar="🤖"):
            print_txt(text)

def record_voice(language="en"):
    """
    Record voice input using speech-to-text
    
    Args:
        language (str, optional): Language for speech recognition. Defaults to "en".
    
    Returns:
        str or None: Recognized text or None
    """
    state = st.session_state
    if "text_received" not in state:
        state.text_received = []
    text = speech_to_text(
        start_prompt="🎤 Click and speak to ask question",
        stop_prompt="⚠️Stop recording",
        language=language,
        use_container_width=True,
        just_once=True,
    )
    if text:
        state.text_received.append(text)
    result = ""
    for text in state.text_received:
        result += text
    state.text_received = []
    return result if result else None

def main():
    """
    Main function to run the voice chatbot
    """
    model = llm_selector()
    
    with st.sidebar:
        question = record_voice(language="en")
    
    # init chat history for a model
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = {}
    if model not in st.session_state.chat_history:
        st.session_state.chat_history[model] = []
    chat_history = st.session_state.chat_history[model]
    
    # print conversation history
    for message in chat_history: 
        print_chat_message(message)
    
    if question:
        user_message = {"role": "user", "content": question}
        print_chat_message(user_message)
        chat_history.append(user_message)
        
        # Generate response using Groq
        answer = generate_groq_response(question, model)
        
        ai_message = {"role": "assistant", "content": answer}
        print_chat_message(ai_message)
        
        # Text-to-speech functionality
        synthesize_speech(answer)
        
        chat_history.append(ai_message)
        
        # truncate chat history to keep 20 messages max
        if len(chat_history) > 20:
            chat_history = chat_history[-20:]
        
        # update chat history
        st.session_state.chat_history[model] = chat_history

if __name__ == "__main__":
    main()
