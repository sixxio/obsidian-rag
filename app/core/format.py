def format_docs(docs):
    return "".join(
        [
            f"Метаданные документа:\n{doc.metadata}\nДокумент:\n{doc.page_content}\n"
            for doc in docs
        ]
    )
