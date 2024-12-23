import streamlit as st
from huggingface_hub import InferenceClient

# Function to clear chat history
def clear_chat_history():
    st.session_state.messages = [{"role": "assistant", "content": "Hello! I'm PolicyPal AI, your personal insurance and finance expert. How may I assist you today?"}]

# Function to predict discount based on fitness score
def predict_discount(fitness_score):
    if fitness_score >= 90:
        return 30
    elif fitness_score >= 80:
        return 25
    elif fitness_score >= 70:
        return 20
    elif fitness_score >= 60:
        return 15
    elif fitness_score >= 50:
        return 10
    elif fitness_score >= 40:
        return 5
    else:
        return 0

# Function to generate AI assistant response
def generate_insurance_assistant_response(prompt_input, client):
    system_message = """You are PolicyPal AI Assistant, created by Akshansh Dwivedi. You are an expert in personal finance and insurance.
    Keep your responses crisp, professional and focused on insurance and financial advice."""

    if "fitness score" in prompt_input.lower() or "discount" in prompt_input.lower():
        return "Please provide your fitness score to get information about the discount you qualify for."

    try:
        user_fitness_score = float(prompt_input)
        discount = predict_discount(user_fitness_score)
        return f"Your fitness score is {user_fitness_score}. Based on this, you qualify for a {discount}% discount on your insurance premium."
    except ValueError:
        pass

    messages = [
        {"role": "system", "content": system_message},
        {"role": "user", "content": prompt_input}
    ]

    response = ""
    for message in client.chat_completion(
            messages=messages,
            max_tokens=512,
            stream=True
    ):
        response += message.choices[0].delta.content or ""

    return response

# Define the AI Assistant page
def ai_assistant_page():
    st.markdown("""
    <style>
        body {
            font-family: 'Arial', sans-serif;
            background-color: #0a192f;
            color: #64ffda;
            line-height: 1.6;
        }
        
        .dashboard-title {
            font-size: 36px;
            font-weight: bold;
            color: #ffffff;
            margin-bottom: 10px;
        }
        
        /* Remove background from chat messages */
        [data-testid="stChatMessage"] div {
            background-color: transparent !important;
        }
        
        /* Chat input container */
        .stChatFloatingInputContainer {
            position: fixed;
            bottom: 0;
            left: 0;
            right: 0;
            background-color: #0a192f;
            padding: 1rem 5rem;
            border-top: 1px solid #172a45;
        }
        
        /* Sidebar styling */
        .sidebar .sidebar-content {
            background-color: #172a45;
            color: #64ffda;
        }
    </style>
    """, unsafe_allow_html=True)

    # Title with custom styling
    st.markdown("<h1 class='dashboard-title'>PolicyPal AI Assistant</h1>", unsafe_allow_html=True)

    # Sidebar configuration
    with st.sidebar:
        st.markdown("<h2 style='color: #ffa500;'>⚙️ Settings</h2>", unsafe_allow_html=True)
        hf_api_token = "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"
        if hf_api_token:
            st.success('API key loaded successfully!', icon='✅')
        else:
            st.error('API key not found', icon='🚨')

        client = InferenceClient(
            "mistralai/Mixtral-8x7B-Instruct-v0.1",
            token=hf_api_token
        )

        if st.button('Clear Chat', help="Clear the chat history"):
            clear_chat_history()

    # Initialize chat history
    if "messages" not in st.session_state:
        st.session_state.messages = [{"role": "assistant", "content": "Hello! I'm PolicyPal AI, your personal insurance and finance expert. How may I assist you today?"}]

    # Display chat messages
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.write(message["content"])

    # Chat input
    if prompt := st.chat_input("Ask about insurance, finance, or enter your fitness score..."):
        st.session_state.messages.append({"role": "user", "content": prompt})
        response = generate_insurance_assistant_response(prompt, client)
        st.session_state.messages.append({"role": "assistant", "content": response})
        st.experimental_rerun()

if __name__ == "__main__":
    ai_assistant_page()
