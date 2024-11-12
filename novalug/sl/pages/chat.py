import streamlit as st

st.title("page 2")

# prompt = st.chat_input("Say something")
# if prompt:
#     st.write(f"User has sent the following prompt: {prompt}")

st.success("This is a success message!", icon="✅")

st.metric("My metric", 42, 2)

# with st.echo():
#     st.write("Code will be executed and printed")

with st.sidebar:
    messages = st.container(height=300)
    if prompt := st.chat_input("Say something"):
        messages.chat_message("user").write(prompt)
        messages.chat_message("assistant").write(f"Echo: {prompt}")
