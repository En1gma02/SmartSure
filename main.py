import streamlit as st

# Set page config at the very top
st.set_page_config(page_title="SmartSure", layout="wide")

from ai_assistant_page import ai_assistant_page
from fitness_score_page import fitness_score_page
from choose_base_plans_page import choose_base_plans_page
from make_your_own_plan_page import make_your_own_plan_page
from home import home_page
#from Bussiness_Dashboard.dashboard import Dashboard  

def main():
    # Add logo at the top of sidebar
    st.sidebar.image("images/SmartSure_icon.png", width=200)
    
    st.sidebar.title("Revolutionizing Insurance with AI")
    
    # Add a brief description
    st.sidebar.markdown("""
    Your personalized insurance journey starts here. 
    Smart solutions for a secure future.
    """)
    
    st.sidebar.markdown("---")
    
    # Initialize session state for default page if not exists
    if 'current_page' not in st.session_state:
        st.session_state['current_page'] = '🏠 Home'
    
    # # Sidebar navigation with emojis
    # page = st.sidebar.selectbox(
    #     "Select a page",
    #     ("🏠 Home", "🤖 AI Assistant", "💪 Fitness Score", "📋 Choose from Base Plans", 
    #      "✨ Make Your Own Plan", "📊 Business Dashboard"),
    #     index=0  # Set default index to 0 (Home)
    # )

     # Sidebar navigation with emojis
    page = st.sidebar.selectbox(
        "Select a page",
        ("🏠 Home", "🤖 AI Assistant", "💪 Fitness Score", "📋 Choose from Base Plans", 
         "✨ Make Your Own Plan"),
        index=0  # Set default index to 0 (Home)
    )
    
    # Remove emoji prefix for page routing
    page = page.split(" ", 1)[1]
    
    if page == "Home":
        home_page()
    elif page == "AI Assistant":
        ai_assistant_page()
    elif page == "Fitness Score":
        fitness_score_page()
    elif page == "Choose from Base Plans":
        choose_base_plans_page()
    elif page == "Make Your Own Plan":
        make_your_own_plan_page()
    #elif page == "Business Dashboard":
        #Dashboard()

if __name__ == "__main__":
    main()
