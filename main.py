import streamlit as st
from llama_index import VectorStoreIndex, ServiceContext, Document
from llama_index.llms import OpenAI
import openai
from llama_index import SimpleDirectoryReader
from llama_index.memory import ChatMemoryBuffer

memory = ChatMemoryBuffer.from_defaults(token_limit=1500)

# input template for Chatbot's personality and task
st.title("AI Chatbot - Instruct your own Custom Chatbot to do anything and provide a knowledge base!")
st.image("https://res.cloudinary.com/preface/image/upload/w_1024,c_limit,f_auto/v1633826301/r2021/assets/images/preface_logo.png",use_column_width=True)

# input template for Chatbot's personality and task
input_template = st.text_area(label="Chatbot Instructions",value=
    """ You are a Chatbot for a Restaurant.
    You are asked to take orders from customers and provide them with the menu.
    Also mention prices of the dishes and the time it will take to prepare the dish (You can estimate using your own logic).
    You can also suggest them the best dishes to order if they ask for recommendations.

    At the end, confirm the order with the customer and ask them if they would like to add anything else to their order.
    """)

# create text input for api key
api_key = st.text_input("Enter your OpenAI API key")

# input pdf file for knowledge base
knowledge_base = st.file_uploader("Upload your knowledge base pdf file")
index = None
# set session state if button is clicked
if knowledge_base:
    st.session_state["knowledge_base"] = knowledge_base

    # save openai key
    openai.api_key = api_key

    # if uploaded save it to knowledge_base.pdf
    if st.session_state.get("knowledge_base"):
        with open("data/knowledge_base.pdf", "wb") as f:
            f.write(st.session_state.get("knowledge_base").getbuffer())
        st.write("File uploaded ✔️")

        # load pdf file and split into chunks
        reader = SimpleDirectoryReader(input_dir="./data", recursive=True)
        docs = reader.load_data()
        service_context = ServiceContext.from_defaults(llm=OpenAI(model="gpt-3.5-turbo", temperature=0.5, system_prompt=input_template))
        index = VectorStoreIndex.from_documents(docs, service_context=service_context)

        st.write("Knowledge base created ✔️")


if st.session_state.get("knowledge_base"):
    st.write("Chatbot Instructions Saved ✔️")

    # Store LLM generated responses
    if "messages" not in st.session_state.keys():
        st.session_state.messages = [{"role": "assistant", "content": "How may I help you?"}]

    # Display chat messages
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.write(message["content"])

    # Function for generating LLM response
    def generate_response(prompt_input):
        # first retrieve the answer from the knowledge base
        chat_engine = index.as_chat_engine(chat_mode="react", memory=memory,verbose=True)

        # # get the answer from the knowledge base
        res = chat_engine.chat(prompt_input)

        return res.response


    # User-provided prompt
    if prompt := st.chat_input():
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.write(prompt)

    # Generate a new response if last message is not from assistant
    if st.session_state.messages[-1]["role"] != "assistant":
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                response = generate_response(prompt)
                st.write(response)
        message = {"role": "assistant", "content": response}
        st.session_state.messages.append(message)