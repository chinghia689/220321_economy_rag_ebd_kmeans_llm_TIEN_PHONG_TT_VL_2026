import re
from typing import Dict, Any

from ingestion.energy_kmeans import EnergyRetriever
from ingestion.model_embedding import vn_embedder
from ingestion.chunks_document import ChromaDBManager
from chatbot.utils.document_grader import DocumentGrader
from chatbot.utils.answer_generator import AnswerGeneratorDocs
from langgraph.graph import END, StateGraph, START
from chatbot.utils.graph_state import GraphState


class FilesChatAgent:
    """
    FilesChatAgent: Tác nhân chatbot sử dụng RAG (Retrieval-Augmented Generation).

    Nhiệm vụ:
        - Nhận câu hỏi người dùng.
        - Truy xuất các tài liệu liên quan từ vector store dùng Energy Retriever.
        - Chấm điểm và lọc ra tài liệu có liên quan.
        - Sinh câu trả lời dựa trên câu hỏi + tài liệu đã lọc.
        - Xử lý trường hợp không tìm thấy câu trả lời.

    Quy trình:
        START → retrieve → grade_documents → (generate | handle_no_answer) → END
    """

    def __init__(self, llm_model, path_vector_store, allowed_files=["*"]) -> None:
        """
        Khởi tạo FilesChatAgent.

        Args:
            llm_model: Mô hình ngôn ngữ (LLM) đã khởi tạo sẵn.
            path_vector_store (str): Đường dẫn đến vector store chứa embeddings.
            allowed_files (list[str], optional): Danh sách file cho phép. Mặc định ["*"].
        """
        self.allowed_files = allowed_files
        self.path_vector_store = path_vector_store

        # Các thành phần xử lý chính
        self.llm = llm_model
        self.document_grader = DocumentGrader(self.llm)
        self.answer_generator = AnswerGeneratorDocs(self.llm)
        
        # Khởi tạo ChromaDB manager để lấy retriever
        self.embeddings = vn_embedder.get_model()
        self.db_manager = ChromaDBManager(
            embeddings_model=self.embeddings,
            persist_dir=path_vector_store
        )
        
        # Lấy vector store từ ChromaDB
        self.vector_store = self.db_manager.vector_store
        if not self.vector_store:
            # Nếu chưa có, load từ disk
            from langchain_chroma import Chroma
            self.vector_store = Chroma(
                persist_directory=path_vector_store,
                embedding_function=self.embeddings
            )
        
        # Khởi tạo Energy Retriever MỘT LẦN (tránh tạo lại mỗi query)
        self.energy_retriever = EnergyRetriever(
            vector_store=self.vector_store,
            embeddings_model=self.embeddings,
            k_retrieve=40,
            similarity_threshold=0.50,
            n_top_clusters=1,
            max_final_docs=15
        )

    def handle_no_answer(self, state: GraphState) -> Dict[str, Any]:
        """
        Xử lý khi không có tài liệu liên quan.

        Args:
            state (GraphState): Trạng thái hiện tại của workflow.

        Returns:
            Dict[str, Any]: Kết quả báo không tìm thấy câu trả lời.
        """
        return {"generation": "❌ Xin lỗi, tôi không tìm thấy thông tin liên quan trong cơ sở dữ liệu để trả lời câu hỏi của bạn."}

    def grade_documents(self, state: GraphState) -> Dict[str, Any]:
        """
        Đánh giá mức độ liên quan của các tài liệu với câu hỏi.
        (Đã tối ưu: Sử dụng Batching gộp N tài liệu vào 1 prompt).

        Args:
            state (GraphState): Trạng thái chứa documents và question.

        Returns:
            Dict[str, Any]: Danh sách tài liệu đã lọc.
        """
        question = state["question"]
        documents = state["documents"]

        print(f"\n📝 Đang chấm điểm hàng loạt {len(documents)} tài liệu...")

        # Gọi hàm grade_batch (chỉ tốn 1 API call duy nhất)
        filtered_docs = self.document_grader.grade_batch(
            question=question, 
            retrieved_docs=documents
        )

        print(f"✅ Đã giữ lại {len(filtered_docs)}/{len(documents)} tài liệu liên quan.")

        return {"documents": filtered_docs, "question": question}
        
    def decide_to_generate(self, state: GraphState) -> str:
        """
        Quyết định có sinh câu trả lời hay không dựa trên số lượng tài liệu lọc.

        Args:
            state (GraphState): Trạng thái chứa documents.

        Returns:
            str: "generate" hoặc "no_document".
        """
        documents = state["documents"]
        
        if not documents:
            print("⚠️ Không có tài liệu liên quan, chuyển sang xử lý không có câu trả lời")
            return "no_document"
        else:
            print("✅ Có tài liệu liên quan, tiến hành sinh câu trả lời")
            return "generate"

    def generate(self, state: GraphState) -> Dict[str, Any]:
        """
        Sinh câu trả lời từ câu hỏi + các tài liệu đã lọc.

        Args:
            state (GraphState): Trạng thái chứa question, documents, prompt.

        Returns:
            Dict[str, Any]: Trả về câu trả lời (generation).
        """
        question = state["question"]
        documents = state["documents"]
        prompt = state.get("prompt", "Bạn là một chuyên gia tư vấn kinh tế.")

        # Ghép nội dung các tài liệu thành context
        context = "\n\n".join(doc.page_content for doc in documents)

        # Sinh câu trả lời từ AnswerGenerator
        generation = self.answer_generator.get_chain().invoke(
            {"question": question, "context": context, "prompt": prompt}
        )

        # Xóa tag <think> nếu có
        generation = re.sub(
            r"<think>.*?</think>", "", generation, flags=re.DOTALL
        ).strip()

        return {"generation": generation}

    def retrieve(self, state: GraphState) -> Dict[str, Any]:
        """
        Truy xuất tài liệu từ vector store dựa trên câu hỏi, sử dụng Energy Retriever.

        Args:
            state (GraphState): Trạng thái chứa câu hỏi.

        Returns:
            Dict[str, Any]: Bao gồm "documents" và "question".
        """
        question = state["question"]

        # Lấy danh sách documents liên quan dùng Energy Distance
        documents = self.energy_retriever.retrieve(query=question)

        return {"documents": documents, "question": question}

    def get_workflow(self):
        """
        Xây dựng workflow xử lý với StateGraph.

        Luồng xử lý:
            START → retrieve → grade_documents
                → (no_document → handle_no_answer | generate) → END

        Returns:
            StateGraph: Workflow đã được định nghĩa.
        """
        workflow = StateGraph(GraphState)

        # Định nghĩa các node
        workflow.add_node("retrieve", self.retrieve)
        workflow.add_node("grade_documents", self.grade_documents)
        workflow.add_node("generate", self.generate)
        workflow.add_node("no_document", self.handle_no_answer)

        # Xây dựng luồng
        workflow.add_edge(START, "retrieve")
        workflow.add_edge("retrieve", "grade_documents")
        
        # Quyết định dựa trên số lượng tài liệu
        workflow.add_conditional_edges(
            "grade_documents",
            self.decide_to_generate,
            {
                "generate": "generate",
                "no_document": "no_document"
            }
        )

        workflow.add_edge("generate", END)
        workflow.add_edge("no_document", END)

        return workflow
