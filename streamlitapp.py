import streamlit as st 
from chatbot import chatbot
st.title("Simple Chatbot App")

# Get user input
user_input1 = st.text_input("Enter your URL:", "")
user_input2 = st.text_input("Enter the question :", "")

    # Check if the user has submitted the inputs
if st.button("Submit"):
        # Concatenate the user inputs and get chatbot response
    chatbot_response=chatbot(user_input1,user_input2)
    # Display the chatbot response
    st.text("Chatbot Response:")
    st.write(chatbot_response)