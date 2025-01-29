from dotenv import load_dotenv

load_dotenv(".env")

from httpx import Client
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_gigachat import GigaChat
from langchain_groq.chat_models import ChatGroq
from langchain_mistralai import ChatMistralAI


from core.format import format_docs
from core.instruction import default_instruction
from core.qdrant import qvs
from core.settings import settings

retriever = qvs.as_retriever(search_type="similarity", search_kwargs={"k": 5})

def get_llm(model_name: str = "ChatGroq"):
    match model_name:
        case 'ChatGroq':
            return ChatGroq(
                model="gemma2-9b-it",
                max_tokens=2048,
                http_client=Client(proxy=str(settings.proxy)),
            )
        case "GigaChat":
            return GigaChat(
                model="GigaChat-Pro",
                credentials=str(settings.giga_api_key),
                timeout=30,
                verify_ssl_certs=False
            )
        case 'MistralAI':
            return ChatMistralAI(model_name='open-mistral-nemo',
                                 temperature=0,
                                 api_key=str(settings.mistral_api_key))
        case _:
            raise ValueError(f"Model {model_name} is not supported.")

def get_rag_chain(llm: str = "ChatGroq", top_k: int = 5, instruction: str = default_instruction):
    retriever.search_kwargs = {"k": top_k}

    template = ChatPromptTemplate(
        [
            (
                "human",
                instruction,
            )
        ]
    )

    return (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | template
        | get_llm(llm)
        | StrOutputParser()
    )
