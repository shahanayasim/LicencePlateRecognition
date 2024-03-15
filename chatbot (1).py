from PyPDF2 import PdfReader
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import FAISS
import os
from langchain.chains.question_answering import load_qa_chain
from langchain.llms import OpenAI

os.environ["OPENAI_API_KEY"] = "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"

# provide the paths of pdf files.
pdf_paths = [
    r"D:\DL_Project\ALPR\vehicle.pdf",
    ]

raw_texts = []

for path in pdf_paths:
    pdf_reader = PdfReader(path)
    raw_text = ''
    for page in pdf_reader.pages:
        content = page.extract_text()
        if content:
            raw_text += content
    raw_texts.append(raw_text)

# We need to split the text using Character Text Split such that it should not increase token size
text_splitter = CharacterTextSplitter(
    separator="\n",
    chunk_size=800,
    chunk_overlap=200,
    length_function=len,
)

texts = [text_splitter.split_text(raw_text) for raw_text in raw_texts]

# Flatten the nested list of texts
flattened_texts = [item for sublist in texts for item in sublist]

# Download embeddings from OpenAI
embeddings = OpenAIEmbeddings()

# Create FAISS index from the flattened texts
document_search = FAISS.from_texts(flattened_texts, embeddings)

chain = load_qa_chain(OpenAI(), chain_type="stuff")


def get_bot_response(user_input):
    bot_response = ""  # Initialize bot_response with an empty string

    if user_input.strip().lower() in ["hi", "hello", "hey", "hy"]:
        bot_response = "Hello, welcome to Legal Chatbot. How can I assist you today!"
    elif user_input.strip().lower() in ["bye", "by", "thank you", "thanks"]:
        bot_response = "bye, and have a good day"
    else:
        question = user_input.strip()  # Extract and clean the question from user input
        if len(question) < 4:
            bot_response = "Please enter a valid question!"
        else:
            docs = document_search.similarity_search(user_input)
            bot_response = chain.run(input_documents=docs, question=question)

    return bot_response


# Example usage:
user_input = input("You: ")
bot_response = get_bot_response(user_input)
print("Bot:", bot_response)
