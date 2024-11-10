from fastapi import FastAPI, Request
import requests
import uvicorn
from langchain.document_loaders import DirectoryLoader, CSVLoader, PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.llms import Ollama
from langchain.vectorstores import FAISS
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.chat_history import BaseChatMessageHistory
from langchain.chains import create_history_aware_retriever
from langchain_huggingface import HuggingFaceEmbeddings

file_directory="/Users/gonzaloguaimas/Documents/code/AI/motoplan-rag/data"
embedding_model='sentence-transformers/all-MiniLM-L6-v2'
phone_id='111'
llm_model ="llama3.2"
embeddings = HuggingFaceEmbeddings(model_name=embedding_model)

app = FastAPI()

VERIFY_TOKEN = 'anytoken'
ACCESS_TOKEN = 'token'

@app.get("/webhook")
async def verify_token(request: Request):
    mode = request.query_params.get('hub.mode')
    token = request.query_params.get('hub.verify_token')
    challenge = request.query_params.get('hub.challenge')

    if mode and token:
        if mode == 'subscribe' and token == VERIFY_TOKEN:
            return int(challenge)
        else:
            return {"error": "Verification token mismatch"}

@app.post("/webhook")
async def receive_message(request: Request):
    try:
        data = await request.json()
        for entry in data.get('entry', []):
            for change in entry.get('changes', []):
                if change.get('field') == 'messages':
                    message = change['value']['messages'][0]
                    # sender_id = message['from']
                    message_text = message['text']['body']
                    chat_bot_response=conversational_rag_chain.invoke(
                        {"input": message_text},
                        config={
                            "configurable": {"session_id": "abc123"}
                        }
                    )
                    print(chat_bot_response["answer"])
                    await send_simple_message('phone', chat_bot_response["answer"])
        return {"status": "success"}
    except Exception as e:
        return {"error": str(e)}
    
@app.post("/send_message")
async def send_message(request: Request):
    data = await request.json()
    recipient_id = data.get('recipient_id')
    message_text = data.get('message_text')
    response = await send_simple_message(recipient_id, message_text)
    return response


async def send_simple_message(to_number: str, message: str):
    url = f"https://graph.facebook.com/v20.0/{phone_id}/messages"
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {ACCESS_TOKEN}"
    }
    payload = {
        "messaging_product": "whatsapp",
        "recipient_type": "individual",
        "to": to_number,
        "type": "text",
        "text": {
            "preview_url": False,
            "body": message
        }
    }
    response = requests.post(url, headers=headers, json=payload)
    return response.json()


#------

def prepare_and_split_docs(directory):
    loaders = [
        DirectoryLoader(directory, glob="**/*.pdf",show_progress=True, loader_cls=PyPDFLoader),
        DirectoryLoader(directory, glob="**/*.csv",loader_cls=CSVLoader)
    ]

    documents=[]
    for loader in loaders:
        data =loader.load()
        documents.extend(data)

    splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
        chunk_size=512, 
        chunk_overlap=256,
        disallowed_special=(),
        separators=["\n\n", "\n", " "]
    )

    split_docs = splitter.split_documents(documents)
    print("Documents splited")
    return split_docs

def ingest_into_vectordb(split_docs):
    db = FAISS.from_documents(split_docs, embeddings)

    DB_FAISS_PATH = 'vectorstore/db_faiss'
    db.save_local(DB_FAISS_PATH)
    print("Documents inserted")
    return db

def get_conversation_chain(retriever):
    llm = Ollama(model=llm_model)
    context_system_prompt = (
        "Teniendo la última conversación en cuenta, contextualizá la pregunta del usuario."
        "Si el usuario necesita una respuesta específica, dá la respuesta en base a los documentos con lo que te entrené. Si te dice: Buen día, ¿cómo estás?, respondé: Buen día, estoy bien, gracias. ¿En qué puedo ayudarte?, es decir respuestas cortas y concisas."
        "No hagás muchas preguntas, siempre tratá de recomendar que valla a una oficina en salta o tucumán o que diga en qué horario se lo puede llamar"
    )

    contextualize_q_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", context_system_prompt),
            MessagesPlaceholder("chat_history"),
            ("human", "{input}"),
        ]
    )
    history_aware_retriever = create_history_aware_retriever(
        llm, retriever, contextualize_q_prompt
    )

    system_prompt = (
        "Sos un asesor comercial de la empresa Motoplan18, tenés que responder de manera tranquila y humana a cada respuesta que te hagan. "
        "Tratá de usar el lenguaje argentino pero no vulgar. Respondé en 1 sóla oración con no más de 15 palabras. "
        "Contestá de manera natural, no te sobreextiendas ni hagas muchas preguntas. No hables en neutro, hablá como argentino. "
        "{context}"
    )

    question_answer_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", system_prompt),
            MessagesPlaceholder("chat_history"),
            ("human", "{input}"),
        ]
    )
    question_answer_chain = create_stuff_documents_chain(llm, question_answer_prompt)
    rag_chain = create_retrieval_chain(history_aware_retriever, question_answer_chain)

    store = {}

    def get_session_history(session_id: str) -> BaseChatMessageHistory:
        if session_id not in store:
            store[session_id] = ChatMessageHistory()
        return store[session_id]

    conversational_rag_chain = RunnableWithMessageHistory(
        rag_chain,
        get_session_history,
        input_messages_key="input",
        history_messages_key="chat_history",
        output_messages_key="answer",
    )
    print("Conversational chain created")
    return conversational_rag_chain


if __name__ == "__main__":
    split_docs=prepare_and_split_docs(file_directory)
    vector_db= ingest_into_vectordb(split_docs)
    retriever =vector_db.as_retriever()
    conversational_rag_chain=get_conversation_chain(retriever)
    uvicorn.run(app, host="0.0.0.0", port=8000)