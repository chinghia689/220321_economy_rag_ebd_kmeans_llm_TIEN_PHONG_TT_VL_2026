import json
import re
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableSequence
from chatbot.utils.custom_prompt import CustomPrompt

class DocumentGrader:
    """
    Lớp kiểm tra HÀNG LOẠT (Batching) xem các documents có liên quan tới câu đầu vào không.
    Giúp giảm số lần gọi API từ 15 lần xuống 1 lần duy nhất.
    """
    def __init__(self, llm) -> None:
        prompt = ChatPromptTemplate.from_messages(
            [
                ("system", CustomPrompt.BATCH_GRADE_DOCUMENT_PROMPT),
                ("human", "Danh sách tài liệu: \n\n {documents} \n\n Câu hỏi: {question}"),
            ]
        )
        self.chain = prompt | llm | StrOutputParser()

    def get_chain(self) -> RunnableSequence:
        return self.chain

    def grade_batch(self, question: str, retrieved_docs: list) -> list:
        if not retrieved_docs:
            return []

        # 1. Gom tất cả documents thành 1 string duy nhất có đánh số
        formatted_docs = "\n".join(
            [f"--- [Tài liệu {i+1}] ---\n{doc.page_content}" for i, doc in enumerate(retrieved_docs)]
        )

        # 2. Gọi LLM đúng 1 lần
        response = self.chain.invoke({
            "documents": formatted_docs,
            "question": question
        })

        # 3. Trích xuất mảng JSON an toàn bằng Regex
        filtered_docs = []
        try:
            match = re.search(r'\[.*?\]', response)
            if match:
                indices = json.loads(match.group(0)) # Chuyển chuỗi "[1, 3]" thành mảng [1, 3]
                for idx in indices:
                    real_idx = idx - 1 # Chuyển index từ (1-15) sang (0-14)
                    if 0 <= real_idx < len(retrieved_docs):
                        filtered_docs.append(retrieved_docs[real_idx])
        except Exception as e:
            print(f"⚠️ Lỗi parse JSON từ LLM: {response}. Lỗi: {e}")

        return filtered_docs