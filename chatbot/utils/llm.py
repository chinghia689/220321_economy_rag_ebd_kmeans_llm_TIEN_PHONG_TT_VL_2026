from langchain_openai import ChatOpenAI
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_groq import ChatGroq
import os  # noqa: E401, F401
# from langchain_xai import ChatXAI


class LLM:
    """
    Lớp tiện ích để khởi tạo các mô hình LLM khác nhau (OpenAI, Gemini, Local/Ollama).

    Thuộc tính:
        temperature (float): Nhiệt độ sinh văn bản (mức độ sáng tạo).
        max_tokens (int): Giới hạn số token tối đa cho đầu ra.
        n_ctx (int): Kích thước context (bộ nhớ ngữ cảnh).
        model (str): Tên mô hình được sử dụng (có thể gán thêm nếu cần).
    """

    def __init__(
        self, temperature: float = 0.01, max_tokens: int = 4096, n_ctx: int = 4096
    ) -> None:
        self.temperature = temperature
        self.n_ctx = n_ctx
        self.max_tokens = max_tokens
        self.model = ""

    def open_ai(self):
        """
        Khởi tạo LLM từ OpenAI (qua API key).

        Yêu cầu biến môi trường:
            - KEY_API_OPENAI
            - OPENAI_LLM_MODEL_NAME

        Returns:
            ChatOpenAI: đối tượng LLM của OpenAI.
        """
        llm = ChatOpenAI(
            openai_api_key=os.environ["KEY_API_OPENAI"],
            model=os.environ["OPENAI_LLM_MODEL_NAME"],
            temperature=self.temperature,
        )
        return llm

    # def local_ai(self):
    #     """
    #     Khởi tạo LLM từ Ollama/local endpoint.

    #     Yêu cầu biến môi trường:
    #         - URL_OLLAMA
    #         - MODEL_CHAT_OLLAMA
    #         - API_KEY_OLLAMA

    #     Returns:
    #         ChatOpenAI: đối tượng LLM giả lập từ local server (Ollama).
    #     """
    #     llm = ChatOpenAI(
    #         base_url=os.environ["URL_OLLAMA"],
    #         model=os.environ["MODEL_CHAT_OLLAMA"],
    #         api_key=os.environ["API_KEY_OLLAMA"],
    #         temperature=self.temperature,
    #     )
    #     return llm

    # def gemini(self):
    #     """
    #     Khởi tạo LLM từ Google Gemini API.

    #     Yêu cầu biến môi trường:
    #         - KEY_API_GOOGLE
    #         - GOOGLE_LLM_MODEL_NAME

    #     Returns:
    #         ChatGoogleGenerativeAI: đối tượng LLM Gemini.
    #     """
    #     llm = ChatGoogleGenerativeAI(
    #         google_api_key=os.environ["GOOGLE_API_KEY"],
    #         model=os.environ["GOOGLE_LLM_MODEL_NAME"],
    #         temperature=self.temperature,
    #     )
    #     return llm

    # def groq(self):
    #     """
    #     Khởi tạo LLM từ Groq API (miễn phí, rate limit cao).

    #     Yêu cầu biến môi trường:
    #         - GROQ_API_KEY

    #     Returns:
    #         ChatGroq: đối tượng LLM Groq.
    #     """
    #     llm = ChatGroq(
    #         api_key=os.environ["GROQ_API_KEY"],
    #         model="llama-3.1-8b-instant",
    #         temperature=self.temperature,
    #     )
    #     return llm

    # def grok(self):
    #     """
    #     Khởi tạo LLM từ Grok (xAI).

    #     Yêu cầu biến môi trường:
    #         - KEY_API_GROK
    #         - GROK_LLM_MODEL_NAME
    #         - URL_GROK (endpoint, thường: https://api.x.ai/v1)
    #     """
    #     return ChatXAI(
    #         api_key=os.environ["KEY_API_GROK"],
    #         model=os.environ["GROK_LLM_MODEL_NAME"],
    #         temperature=self.temperature,
    #     )

    def get_llm(self, llm_name: str):
        """
        Lấy mô hình LLM dựa trên tên.

        Args:
            llm_name (str): Tên mô hình mong muốn.
                            Các lựa chọn: "openai", "gemini", "local".

        Returns:
            Any: Đối tượng LLM tương ứng (OpenAI/Gemini/Local).
        """
        if llm_name == "openai":
            return self.open_ai()

        # if llm_name == "gemini":
        #     return self.gemini()

        # if llm_name == "groq":
        #     return self.groq()

        # if llm_name == "local":
        #     return self.local_ai()

        # if llm_name == "grok":
        #     return self.grok()

        raise ValueError(f"LLM '{llm_name}' không hỗ trợ. Chọn: gemini, groq")
