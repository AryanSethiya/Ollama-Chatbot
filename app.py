import streamlit as st
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.llms import Ollama
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# LangSmith tracking (optional but useful for debugging)
os.environ["LANGCHAIN_API_KEY"] = os.getenv("LANGCHAIN_API_KEY")
os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_PROJECT"] = "Simple Q&A Chatbot With Ollama"

# Prompt template
prompt = ChatPromptTemplate.from_messages(
    [
        ("system", "You are a helpful assistant. Please respond to the user's queries."),
        ("user", "Question: {question}")
    ]
)

# Function to generate response
def generate_response(question, model_name, temperature, max_tokens):
    try:
        llm = Ollama(model=model_name, temperature=temperature)
        output_parser = StrOutputParser()
        chain = prompt | llm | output_parser
        answer = chain.invoke({'question': question})
        return answer
    except Exception as e:
        return f"🚨 Error: {e}"

# Streamlit UI
st.title("Enhanced Q&A Chatbot with Ollama")

# Sidebar - model selection
selected_model = st.sidebar.selectbox("Select Open Source Model", ["llama3.2", "gemma3:1b"])

# Sidebar - parameters
temperature = st.sidebar.slider("Temperature", min_value=0.0, max_value=1.0, value=0.7)
max_tokens = st.sidebar.slider("Max Tokens", min_value=50, max_value=300, value=150)

# Main input
st.write("Go ahead and ask any question:")
user_input = st.text_input("You:")

if user_input:
    response = generate_response(user_input, selected_model, temperature, max_tokens)
    st.write("💬 Response:")
    st.write(response)
else:
    st.write("📌 Please enter a question in the input box above.")
