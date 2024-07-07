import streamlit as st
from langchain_community.document_loaders.csv_loader import CSVLoader
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings
from langchain_core.prompts import PromptTemplate
from langchain_openai import ChatOpenAI
from langchain.chains import LLMChain
# from langchain import RunnableSequence
from dotenv import load_dotenv
import os

# Set the page configuration at the top of the script
st.set_page_config(page_title="CustomerCareBot", page_icon="🤖")

load_dotenv()

os.environ['OPENAI_API_KEY'] =os.getenv('OPENAI_API_KEY')

# Path for the FAISS index
FAISS_INDEX_PATH = "faiss_index"

# Load or create FAISS index
def load_or_create_faiss_index():
    embeddings = OpenAIEmbeddings()
    if os.path.exists(FAISS_INDEX_PATH):
        db = FAISS.load_local(FAISS_INDEX_PATH, embeddings, allow_dangerous_deserialization = True)
        st.info("Loaded vectors from FAISS index.")
    else:
        # Vectorise the sales response csv data
        loader = CSVLoader(file_path="Customer_Service_Assistants.csv")
        documents = loader.load()
        db = FAISS.from_documents(documents, embeddings)
        db.save_local(FAISS_INDEX_PATH)
        st.info("Created and saved vectors to FAISS index.") 
    return db

db = load_or_create_faiss_index()

# Initialize conversation history in session state
if "conversation_history" not in st.session_state:
    st.session_state["conversation_history"] = []

# Initialize the text area input in session state
if "input_message" not in st.session_state:
    st.session_state["input_message"] = ""

# 2. Function to retrieve similar responses
def retrieve_info(query):
    similar_response = db.similarity_search(query, k=3)
    page_contents_array = [doc.page_content for doc in similar_response]
    # print(page_contents_array)

    return page_contents_array


# 3. Setup LLMChain & prompts
llm = ChatOpenAI(temperature=0.7, model="gpt-3.5-turbo")

template = """
You are a world class business development representative. 
I will share a prospect's message with you and you will give me the best answer that 
I should send to this prospect based on past best practies, 
and you will follow ALL of the rules below:

1/ Response should be very similar or even identical to the past best practies, 
in terms of length, tone of voice, logical arguments and other details

2/ If the best practice are irrelevant, then try to mimic the style of the best practice to prospect's message

Below is a message I received from the prospect:
{message}

Here is a list of best practies of how we normally respond to prospect in similar scenarios:
{best_practice}

Please write the best response that I should send to this prospect:
"""

prompt = PromptTemplate(
    input_variables=["message", "best_practice"],
    template=template
)

chain = LLMChain(llm=llm, prompt=prompt)

# Function to update conversation history
def update_conversation_history(user_message, bot_response):
    # Limit the history to the last 5 messages (adjust as needed)
    st.session_state["conversation_history"].append({"role": "user", "content": user_message})
    st.session_state["conversation_history"].append({"role": "assistant", "content": bot_response})
    # if len(conversation_history) > 10:
    #     conversation_history.pop(0)
    #     conversation_history.pop(0)

def get_contextual_input(conversation_history, new_message):
    history_str = ""
    for entry in conversation_history:
        role = entry["role"]
        content = entry["content"]
        history_str += f"{role}: {content}\n"
    history_str += f"user: {new_message}\n"
    return history_str

# 4. Retrieval augmented generation
def generate_response(message):
    contextual_input = get_contextual_input(st.session_state["conversation_history"], message)
    best_practice = retrieve_info(contextual_input)
    response = chain.run(message=contextual_input, best_practice=best_practice)
    update_conversation_history(message, response)
    return response

# Function to display conversation history
def display_conversation_history():
    # for entry in reversed(st.session_state["conversation_history"]):
    #     role = "User" if entry["role"] == "user" else "Assistant"
    #     st.info(f"**{role}:** {entry['content']}")
    #     st.write("------------------------------------------------")
    
    # Access the conversation history from the session state
    history = st.session_state["conversation_history"]
    for i in range(len(history) - 1, -1, -2):
        # Ensure there is a pair of messages (user and assistant) to display
        if i - 1 >= 0:
            # Get the user message and assistant response
            user_entry = history[i - 1]
            bot_entry = history[i]
            # Display the user and bot message with a header "User:" and "Assistant:"
            st.info(f"**User:** {user_entry['content']}")
            st.write("------------------------------------------------")
            st.info(f"**Assistant:** {bot_entry['content']}")

# 5. Build the app with streamlit
def main():
    # st.set_page_config(page_title="CustomerCareBot", page_icon="🤖")
    st.header("CustomerCareBot 🤖")

    # Text area for user input
    st.session_state["input_message"] = st.text_area("Customer message", st.session_state["input_message"] )
    
    if st.session_state["input_message"] :
        with st.spinner("Generating best practice message..."):
            try:
                result = generate_response(st.session_state["input_message"] )
                st.session_state["input_message"] = ""
                # st.info(result)
            except Exception as e:
                st.error(f"Error generating best practice message: {e}")

    # Display conversation history
    display_conversation_history()

if __name__ == '__main__':
    main()