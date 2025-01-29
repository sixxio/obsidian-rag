import re

import requests
import streamlit as st
from core.instruction import instructions
from core.settings import settings


def is_valid_instruction(instruction: str) -> bool:
    return (
        re.fullmatch(
            r".*\{context\}.*\{question\}.*|.*\{question\}.*\{context\}.*", instruction, re.S
        )
        is not None
    )

st.set_page_config(page_title='Obsiminers',
                   page_icon=':robot_face:')

ask_tab, add_tab = st.tabs(["Задать вопрос", "Добавить документ в коллекцию"])

with ask_tab:
    question = st.text_input("Ваш вопрос")

    model_choice = st.selectbox("Выберите модель:", ["ChatGroq", "GigaChat", "MistralAI"])


    top_k = st.slider("Количество результатов (top_k):", min_value=1, max_value=10, value=5)

    instruction_type = st.selectbox("Выберите системный промпт", list(instructions) + ["Кастом"], 0)

    if instruction_type == "Кастом":
        custom_instruction = st.text_area(
            "Кастомная инструкция",
            placeholder="Введите свою инструкцию здесь\n\nНе забудьте указать поля {question} и {context}",
            height=200,
        )

    with st.expander("Системные промпты"):
        for i in instructions:
            st.markdown(f"##### {i}")
            st.markdown(instructions[i])

    if st.button("Найти"):
        if question:
            if instruction_type == "Кастом" and not is_valid_instruction(custom_instruction):
                st.warning(
                    "Кастомная инструкция должна иметь поля {question} и {context}, порядок полей в инструкции неважен."
                )

            else:
                instruction = (
                    custom_instruction
                    if instruction_type == "Кастом"
                    else instructions[instruction_type]
                )

                try:
                    ans = requests.get(
                        str(settings.fastapi_host) + "ask",
                        params={
                            "user_question": question,
                            "model": model_choice,
                            "top_k": top_k,
                            "instruction": instruction,
                        },
                        timeout=10,
                    ).json()["answer"]

                    st.markdown(ans)
                except Exception as e:
                    st.warning(f"Бэкенд еще не проснулся или с ним что-то не так!\n{e}")
        else:
            st.warning("Вы забыли ввести свой вопрос!")

with add_tab:
    files = st.file_uploader("Выберите файлы", accept_multiple_files=True, type="md")

    if st.button("Загрузить"):
        payload = [{"path": file.name, "text": file.read().decode("utf-8")} for file in files]

        requests.post(str(settings.fastapi_host) + "update", json=payload, timeout=10)

        files_num = len(payload)

        st.success(
            f'Успешно загружен{"ы" if files_num > 1 else ""} {files_num} {"документов" if files_num > 4 else "документа" if files_num > 1 else "документ"}.'
        )
