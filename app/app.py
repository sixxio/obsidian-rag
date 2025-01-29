from typing import Annotated, List

import mlflow
import uvicorn
from api.schemas import Document, LLMAnswer
from core.chain import get_rag_chain
from core.instruction import default_instruction
from core.settings import settings
from core.update import update_collection
from fastapi import FastAPI, HTTPException, Query

mlflow.set_experiment("Obsiminers Tracing")
mlflow.langchain.autolog()

app = FastAPI()

@app.get("/ask", response_model=LLMAnswer)
async def generate_answer(
    user_question: Annotated[str, Query(description="User's question")],
    model: Annotated[str, Query(description="Model Instance to use for generating the answer")] = "ChatGroq",
    top_k: Annotated[int, Query(ge=1, le=10, description="Number of top results to retrieve (1-10)")] = 5,
    instruction: Annotated[str, Query(description="Custom instruction for the model")] = default_instruction
):
    try:
        rag_chain = get_rag_chain(model, top_k=top_k, instruction=instruction)
        answer_text = rag_chain.invoke(user_question, verbose=True)
        return LLMAnswer(answer=answer_text)
    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Error generating answer: {e!s}"
        ) from e


@app.post("/update")
def update_database(docs: List[Document]):
    for doc in docs:
        update_collection(doc.path, doc.text)


if __name__ == "__main__":
    uvicorn.run(app, host=settings.fastapi_host.host, port=settings.fastapi_host.port) # type: ignore
