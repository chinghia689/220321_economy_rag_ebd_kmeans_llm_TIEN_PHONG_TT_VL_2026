"""
Main entry point cho Chatbot RAG với Energy Distance Retriever.

Một ứng dụng chatbot sử dụng:
- Vector Database (Chroma) với embeddings tiếng Việt
- Energy-Based Distance cho retrieval nâng cao
- LangGraph cho workflow xử lý
- LLM (OpenAI, Gemini, local Ollama, etc.) để sinh câu trả lời
"""

import os
import sys
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

# Thêm parent folder vào path
sys.path.insert(0, str(Path(__file__).parent.parent))

from chatbot.utils.llm import LLM
from chatbot.services.files_rag_chat_agent import FilesChatAgent
from chatbot.utils.graph_state import GraphState


class ChatbotRunner:
    """
    Runner chính để chạy chatbot ứng dụng.
    """

    def __init__(self, path_vector_store, llm_provider="gemini"):
        """
        Khởi tạo Chatbot.

        Args:
            path_vector_store (str): Đường dẫn đến vector store (VD: './chroma_cosine')
            llm_provider (str): Loại LLM (openai, gemini, local, grok). Default: gemini
        """
        self.path_vector_store = path_vector_store
        self.llm_provider = llm_provider

        # Khởi tạo LLM
        llm_handler = LLM()
        self.llm = llm_handler.get_llm(llm_provider)

        # Khởi tạo Chatbot Agent
        self.agent = FilesChatAgent(
            llm_model=self.llm,
            path_vector_store=path_vector_store
        )

        # Xây dựng workflow
        self.workflow = self.agent.get_workflow()
        self.compiled_workflow = self.workflow.compile()

    def answer_question(self, question: str, prompt: str = None) -> str:
        """
        Trả lời câu hỏi của người dùng.

        Args:
            question (str): Câu hỏi của người dùng
            prompt (str, optional): Custom prompt hệ thống. Nếu None, dùng mặc định.

        Returns:
            str: Câu trả lời từ chatbot
        """
        if not prompt:
            prompt = """Bạn là một chuyên gia tư vấn kinh tế Việt Nam.
Hãy trả lời câu hỏi CHỈ dựa trên thông tin trong ngữ cảnh được cung cấp.
Nếu ngữ cảnh không chứa thông tin cần thiết, hãy nói rõ là không có thông tin."""

        print(f"\n{'='*60}")
        print(f"📝 Câu hỏi: {question}")
        print(f"{'='*60}")

        # Chuẩn bị input state
        input_state = {
            "question": question,
            "generation": "",
            "documents": [],
            "prompt": prompt
        }

        # Chạy workflow
        output_state = self.compiled_workflow.invoke(input_state)

        # Lấy kết quả
        answer = output_state.get("generation", "❌ Không thể tạo câu trả lời.")

        print(f"\n💭 Câu trả lời:")
        print(f"{'-'*60}")
        print(answer)
        print(f"{'-'*60}\n")

        return answer

    def interactive_chat(self):
        """
        Chế độ chat tương tác với người dùng.
        """
        print("\n" + "="*60)
        print("🤖 CHATBOT KINH TẾ - INTERACTIVE MODE")
        print("="*60)
        print("📌 Gõ 'exit' hoặc 'quit' để thoát")
        print("Các tùy chọn lệnh:")
        print("  - /custom_prompt <text>  : Đặt custom prompt")
        print("  - /clear                 : Xóa prompt về mặc định")
        print("="*60 + "\n")

        custom_prompt = None

        while True:
            question = input("❓ Nhập câu hỏi: ").strip()

            if question.lower() in ["exit", "quit"]:
                print("👋 Cảm ơn bạn đã sử dụng chatbot. Tạm biệt!")
                break

            if question.lower().startswith("/custom_prompt"):
                custom_prompt = question.replace("/custom_prompt", "").strip()
                print(f"✅ Custom prompt đã được đặt: {custom_prompt}\n")
                continue

            if question.lower() == "/clear":
                custom_prompt = None
                print("✅ Custom prompt đã được xóa, quay lại mặc định.\n")
                continue

            if not question:
                print("⚠️ Vui lòng nhập một câu hỏi hợp lệ.\n")
                continue

            # Gọi assistant để trả lời
            self.answer_question(question, prompt=custom_prompt)


def main():
    """
    Hàm main - điểm bắt đầu chương trình.
    """
    # Cấu hình mặc định
    VECTOR_STORE_PATH = "./chroma_economy_db"  # Thay đổi theo đường dẫn thực tế
    LLM_PROVIDER = "openai"  # options: gemini, groq (groq miễn phí, rate limit cao)

    # Kiểm tra xem vector store có tồn tại không
    if not os.path.exists(VECTOR_STORE_PATH):
        print(f"❌ Lỗi: Vector store không tìm thấy tại '{VECTOR_STORE_PATH}'")
        print(f"💡 Vui lòng chạy ingestion trước bằng: python ingestion/vector_data_builder.py")
        sys.exit(1)

    print("🚀 Đang khởi tạo chatbot...")

    # Khởi tạo chatbot runner
    chatbot = ChatbotRunner(
        path_vector_store=VECTOR_STORE_PATH,
        llm_provider=LLM_PROVIDER
    )

    # Demo nhanh hoặc chế độ interactive
    import argparse
    parser = argparse.ArgumentParser(description="Chatbot RAG Demo")
    parser.add_argument("--question", type=str, help="Câu hỏi trực tiếp (không cần interactive)")
    parser.add_argument("--llm", type=str, default="gemini", help="LLM provider (gemini, openai, local, grok)")
    args = parser.parse_args()

    # Nếu có câu hỏi trực tiếp
    if args.question:
        chatbot.answer_question(args.question)
    else:
        # Chế độ interactive
        chatbot.interactive_chat()


if __name__ == "__main__":
    main()
