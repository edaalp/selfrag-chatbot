import os, chromadb, re, torch, asyncio, warnings, torch
from langchain_community.embeddings import SentenceTransformerEmbeddings
from langchain_community.chat_models import ChatOllama
from langchain_community.vectorstores import Chroma
from langchain.prompts import PromptTemplate, ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser, StrOutputParser
from langchain_core.pydantic_v1 import BaseModel, Field
from langchain import hub
import chainlit as cl
from langchain_core.prompts import ChatPromptTemplate
warnings.filterwarnings("ignore")

os.environ["LANGCHAIN_API_KEY"] = "YOUR_API_KEY"
cl.run_sync = False
documents = []
question = ""
llm = ChatOllama(model="gemma2:2b", format="json", temperature=0)
rag_prompt = hub.pull("rlm/rag-prompt")

@cl.on_chat_start
async def initialize():
    
    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "You're an assistant that can reach data from documents and answer questions with these data.",
            ),
            
        ]
    )
    runnable = prompt | llm | StrOutputParser()
    cl.user_session.set("runnable", runnable)
    data_path ="data\\"
    documents = []
    for filename in os.listdir(data_path):
        file_path = data_path+ "\\" + filename
            # Open and read the file
        with open(file_path, encoding="utf8") as file:
                file_contents = file.read()
                documents.append(file_contents)


    #splitting the texts
    def text_splitter(texts):
        chunks =[]
        sentence_ending = "#SECTION_END#"
        sentence_start = "#SECTION_START#"
        sentence_topic = "#SECTION_TOPIC: "
        for text in texts:
        
            splits = re.split(f"{re.escape(sentence_start)}|{re.escape(sentence_ending)}", text)
            
            # Remove any empty strings from the result
            splits = [s for s in splits if s.strip()]
            chunks.extend(splits)
        return chunks
    
    documents = text_splitter(documents)
    #adding to vector database

    DATABASE_PATH = "embeddings\\"
    client =  chromadb.Client()
    collection = client.get_or_create_collection(name= "embeddings")
    embeddings = SentenceTransformerEmbeddings(model_name = "sentence-transformers/all-MiniLM-L6-v2")
    vectorstore  = await Chroma.afrom_texts(
            collection_name= "embeddings",
            texts = documents,
            embedding=embeddings,
            client= client,
            persist_directory= DATABASE_PATH
    )
    global retriever
    retriever = vectorstore.as_retriever(kwargs={"k": 1})

    await cl.Message("Merhaba!").send()


@cl.on_message
async def main(message: cl.Message):
    lastInput = ""
    question = message.content
    
    async def grade_docs(documents, question):
        class GradeDocuments(BaseModel):
            """Binary score for relevance check on retrieved documents."""

            binary_score: str = Field(
                description="Documents are relevant to the question, 'yes' or 'no'"
            )
        llm = ChatOllama(model="gemma2:2b", format="json", temperature=0)
        
        prompt_grade = PromptTemplate(
            template="""You are a grader assessing relevance of a retrieved document to a user question. \n 
            Here is the retrieved document: \n\n {document} \n\n
            Here is the user question: {question} \n
            If the document contains keywords related to the user question, grade it as relevant. \n
            It does not need to be a stringent test. The goal is to filter out erroneous retrievals. \n
            Give a binary score 'yes' or 'no' score to indicate whether the document is relevant to the question. \n
            Provide the binary score as a JSON with a single key 'score' and no premable or explanation. For generating sentences, only use Turkish.""",
            input_variables=["question", "document"]
        )

        retrieval_grader = prompt_grade | llm | JsonOutputParser()
        documents = retriever.get_relevant_documents(question)
        doc_txt = documents[1]
        result =  await retrieval_grader.ainvoke({"question": question, "document": doc_txt})
        return result

    # LLM
    async def generate_answer(documents, question): 
        prompt_generate = rag_prompt
        prompt_generate += "Only answer in Turkish. DO NOT answer in english."

        llm = ChatOllama(model="gemma2:2b", temperature=0, )

        # Chain
        rag_chain = prompt_generate | llm | StrOutputParser()

        # Run
        generation = rag_chain.invoke({"context": documents, "question": question})
        return generation


    #hallucination grader

    ### Hallucination Grader
    async def hallucination_grader(documents, generation):
        # Prompt
        prompt_hallucination = PromptTemplate(
            template="""You are a grader assessing whether an answer is grounded in / supported by a set of facts. \n 
            Here are the facts:
            \n ------- \n
            {documents} 
            \n ------- \n
            Here is the answer: {generation}
            Give a binary score 'yes' or 'no' score to indicate whether the answer is grounded in / supported by a set of facts. \n
            Provide the binary score as a JSON with a single key 'score' and no preamble or explanation.""",
            input_variables=["generation", "documents"],
        )

        hallucination_grader = prompt_hallucination | llm | JsonOutputParser()
        return await hallucination_grader.ainvoke({"documents": documents, "generation": generation})
        

    ### Answer Grader
    async def grade_answer(question, generation): 
        class GradeAnswer(BaseModel):
            """Binary score to assess answer addresses question."""

            binary_score: str = Field(
                description="Answer addresses the question, 'yes' or 'no', do not give an empty answer"
            )

        # Prompt
        system = """You are a grader assessing whether an answer addresses / resolves a question \n
            Give a binary score 'yes' or 'no'. Yes' means that the answer resolves the question. Do not add new lines"""
        answer_prompt = ChatPromptTemplate.from_messages(
            [
                ("system", system),
                ("human", "User question: \n\n {question} \n\n LLM generation: {generation}"),
            ]
        )

        answer_grader = answer_prompt | llm
        answer_grade = await answer_grader.ainvoke({"question": question, "generation": generation})
        return answer_grade


    ### Question Re-writer 
    #This is NOT used, I just left it here because it works. This is not used because creating the flow as it was in the paper would make the code run reallly slow in my machine.
    #My code does not try to generate in every new sentence but creates the answer once.
    async def rewrite_question(question):
        # Prompt
        system = """You a question re-writer that converts an input question to a better version that is optimized \n
            for vectorstore retrieval. Look at the input and try to reason about the underlying semantic intent / meaning."""
        re_write_prompt = ChatPromptTemplate.from_messages(
            [
                ("system", system),
                (
                    "human",
                    "Here is the initial question: \n\n {question} \n Formulate an improved question in Turkish language.",
                ),
            ]
        )

        question_rewriter = re_write_prompt | llm | StrOutputParser()
        return (await question_rewriter.ainvoke({"question": question}))
    
    async def need_retrieval(question): 
        class NeedRetrieval(BaseModel):
            """Binary score to assess if any factual information is needed to answer the question"""

            binary_score: str = Field(
                description="Answer addresses the question, 'yes' or 'no', do not give an empty answer"
            )

        # Prompt
        system = """Given an instruction, please make a judgment on whether finding some external documents
        from the web (e.g., Wikipedia) helps to generate a better response. Give a binary score "yes" or "no". "yes" indicated that external document would be helpful. Provide the binary score as a JSON with a single key 'score' and no preamble or explanation. """
        need_prompt = ChatPromptTemplate.from_messages(
            [
                ("system", system),
                ("human", "User question: \n\n {question}"),
            ]
        )

        need_grader = need_prompt | llm
        need_grade = await need_grader.ainvoke({"question": question})
        return need_grade
    


    #Main code to execute. The flow can be seen here.
    need_of_retrieval = await need_retrieval(question)
    print(need_of_retrieval.content)
    if """{"score": "yes"}""" in need_of_retrieval.content:
        documents = retriever.get_relevant_documents(question)
        print("Getting relevant documents\n")
        gradings = []
        tasks = [asyncio.create_task(grade_docs(documents[i], question=question)) for i in range(len(documents))]
        await asyncio.wait(tasks)
        gradings = [task.result() for task in tasks]
        relevant_exists = False
        for i in gradings:
            if i == {'score': 'yes'} :
                relevant_exists =True

        if not relevant_exists:
            lastInput = "There is no relevant document"
            print(lastInput)
            await cl.Message("Bu konuyla ilgili bir bilgim yok.").send()
        else:
            print(f"Grading output: {gradings}")

            answers = [len(gradings)]
            hallucinations = [len(gradings)]
            tasks1 = [
                asyncio.create_task(generate_answer(documents=documents[i], question=question)) 
                for i in range(len(gradings)) 
                if gradings[i] == {'score': 'yes'}
            ]    
            await asyncio.wait(tasks1)
            answers = [task.result() for task in tasks1]
            if not answers:
                lastInput = ""
            else:
                print(answers)
            tasks2 =[ asyncio.create_task(hallucination_grader(documents=documents[i], generation  = answers[i])) for i in range(len(gradings))]
            await asyncio.wait(tasks2)
            hallucinations = [task.result() for task in tasks2]
            hallucination_exists = False
            for i in hallucinations:
                if i == {'score': 'yes'} :
                    hallucination_exists =True

            if not hallucination_exists:
                lastInput = "There is no non-halucinating generation."
                print(lastInput)
            else:
                print("No Hallucinations: " , hallucinations)
            
            
            for i in range(len(answers)):
                if(hallucinations[i] == {'score': 'yes'} ): # this means answer does not have hallucination
                    lastInput += answers[i]

            lastInput = lastInput.replace("\n" , "")
        
            if lastInput == "There is no non-halucinating generation.":
                await cl.Message("Bu sorunun cevabı belgelerde geçmiyor.").send()
            else: 
                lastInput = lastInput[:200]
                print("Final generation...")
                await cl.Message(await generate_answer(lastInput, question)).send()
    else:
        last_message  ="Lütfen sorunuzu sorun."
        await cl.Message(last_message).send()
