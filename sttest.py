import streamlit as st


st.title("Chat with LLM")
st.caption('test')



st.write('hi')

data = 'foo'

if st.checkbox('Show raw data'):
    st.subheader('Raw data')
    st.write(data)


st.slider('select',1,2,3)





prompt = st.chat_input("Say something")


if prompt:
    mess = st.chat_message("user")
    mess.write(prompt)


    message = st.chat_message("assistant")
    message.write("Hello human")


with st.sidebar:
        openai_api_key = st.text_input("OpenAI API Key", key="chatbot_api_key", type="password")
        "[Get an OpenAI API key](https://platform.openai.com/account/api-keys)"
        "[View the source code](https://github.com/streamlit/llm-examples/blob/main/Chatbot.py)"
        "[![Open in GitHub Codespaces](https://github.com/codespaces/badge.svg)](https://codespaces.new/streamlit/llm-examples?quickstart=1)"
