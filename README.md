# Overview
This application is designed to enable a chatbot-like assistant that retrieves relevant information from a set of documents and generates accurate, fact-based answers to user queries. The system leverages LangChain and Chainlit to process and manage documents, perform information retrieval, and generate answers using a pre-trained language model. It aims to provide answers in Turkish based on the retrieved information, while also performing grading to ensure the relevance and accuracy of the generated responses.

# Features:
- Retrieves relevant documents based on a user's query.
- Grades document relevance, ensuring that only relevant information is used for answering questions.
- Generates answers from retrieved documents, with checks to prevent hallucinations.

# Setup Environment:
Before running the application, you must set your LangChain API Key as an environment variable.
Alternatively, you can directly add it in the code where os.environ["LANGCHAIN_API_KEY"] is set.

Your text/pdf/word files must be inside the data folder. The more data you have, more relevant generations may be possible. If there is no relevant information in the documents that you provide, chatbot will respond indicating that it has no information.

If there exists relevant information but the generations are hallucinated, the bot will again not answer in order not to provide wrong information.

# How to Run?
Run a terminal inside the file location (or simply cd <file_path>). Run the line
`chainlit run app.py`
