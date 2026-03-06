from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableSequence
from langchain_core.output_parsers import StrOutputParser

# from app.config import settings
from chatbot.utils.custom_prompt import CustomPrompt


class AnswerGeneratorDocs:
    """
    Lớp AnswerGeneratorDocs:
        - Sinh câu trả lời từ câu hỏi và tập context (RAG).
        - Prompt bao gồm hướng dẫn hệ thống, prompt bổ sung, và context.
    """

    def __init__(self, llm) -> None:
        prompt = ChatPromptTemplate.from_messages(
            [
                ("system", CustomPrompt.GENERATE_ANSWER_PROMPT),
                # ("system", "{prompt}"),
                ("human", """Ngữ cảnh (CHỈ sử dụng thông tin trong đây):
---
{context}
---

Câu hỏi: {question}"""),
            ]
        )
        self.chain = prompt | llm | StrOutputParser()

    def get_chain(self) -> RunnableSequence:
        """Trả về chain sinh câu trả lời dựa trên question + context."""
        return self.chain
