from app.config import settings
import os
import ast
from chatbot.utils.create_file_data import save_to_excel
import json
import pandas as pd
from datasets import Dataset

# Metrics từ Ragas
from ragas.metrics import context_precision, context_recall, faithfulness, answer_relevancy, answer_correctness

try:
    from ragas.metrics import answer_correctness as _answer_correctness

    HAS_CORRECTNESS = True
except Exception:
    HAS_CORRECTNESS = False

from ragas import evaluate
from ragas.run_config import RunConfig
from langchain_openai import ChatOpenAI, OpenAIEmbeddings


def _to_context_list(x):
    """
    Ép cột contexts (trong file Excel) về list[str].

    Hỗ trợ các định dạng đầu vào:
        - List gốc (Python list).
        - NaN (trả về []).
        - Chuỗi dạng JSON / literal (ví dụ "['a','b']").
        - Chuỗi phân tách bởi "||" hoặc xuống dòng "\n".
        - Chuỗi đơn lẻ.

    Args:
        x (Any): giá trị từ DataFrame (cột contexts).

    Returns:
        list[str]: danh sách context đã chuẩn hoá.
    """
    if isinstance(x, list):
        return [str(i) for i in x]
    if pd.isna(x):
        return []
    s = str(x).strip()

    # Thử parse bằng JSON hoặc ast.literal_eval
    for parser in (json.loads, ast.literal_eval):
        try:
            v = parser(s)
            if isinstance(v, list):
                return [str(i) for i in v]
            if isinstance(v, dict):
                return [str(vv) for vv in v.values()]
        except Exception:
            pass

    # Thử parse bằng "||"
    if "||" in s:
        return [i.strip() for i in s.split("||") if i.strip()]

    # Thử parse bằng xuống dòng
    if "\n" in s:
        return [i.strip() for i in s.splitlines() if i.strip()]

    return [s] if s else []


def score_excel_with_ragas_to_xlsx(
    excel_path: str,
    output_xlsx: str = "ragas_scores.xlsx",
    question_column: str = "question",
    answer_column: str = "answer",
    contexts_column: str = "contexts_answer",
    ground_truth_column: str = "ground_truth",
    # cấu hình Ragas:
    ragas_batch_size: int = 2,  # batch nội bộ (khuyến nghị 2–8)
    ragas_max_workers: int = 1,  # số job chạy song song
    ragas_timeout: int = 360,  # timeout mỗi job (giây)
    ragas_max_retries: int = 3,  # số lần retry khi lỗi tạm thời
):
    """
    Chấm điểm các câu hỏi–trả lời trong file Excel bằng bộ metrics của Ragas.

    File Excel input cần có các cột:
        - question
        - answer
        - contexts_answer
        - ground_truth (tuỳ chọn, để tính answer_correctness)

    Metrics dùng (chậm → nhanh):
        - faithfulness  (***)   : Trung thực với ngữ cảnh (phạt bịa/hallucination).
        - answer_relevancy (**) : Liên quan giữa answer và question.
        - context_precision (*) : Tỷ lệ context đúng.
        - context_recall (*)    : Độ bao phủ context.
        - answer_correctness (**) [tuỳ chọn] : So sánh answer với ground_truth.

    Mẹo tăng tốc:
        - Giảm batch_size hoặc max_workers.
        - Tạm tắt faithfulness khi debug.

    Args:
        excel_path (str): Đường dẫn file Excel input.
        output_xlsx (str): File Excel output.
        question_column (str): Tên cột question.
        answer_column (str): Tên cột answer.
        contexts_column (str): Tên cột contexts.
        ground_truth_column (str): Tên cột ground_truth.
        ragas_batch_size (int): Batch size cho Ragas.
        ragas_max_workers (int): Số worker song song.
        ragas_timeout (int): Timeout cho mỗi job (giây).
        ragas_max_retries (int): Số lần retry khi lỗi tạm thời.

    Returns:
        str: Đường dẫn file Excel kết quả.
    """
    # 1. Đọc file Excel
    df = pd.read_excel(excel_path)

    # 2. Kiểm tra cột bắt buộc
    for col in [question_column, answer_column, contexts_column]:
        if col not in df.columns:
            raise ValueError(f"Thiếu cột '{col}' trong file Excel.")

    # 3. Chuẩn hoá contexts
    df = df.copy()
    contexts = df[contexts_column].apply(_to_context_list).tolist()

    # 4. Chuẩn bị Dataset cho Ragas
    ragas_cols = {
        "question": df[question_column].astype(str).tolist(),
        "answer": df[answer_column].astype(str).tolist(),
        "contexts": contexts,
    }

    use_correctness = HAS_CORRECTNESS and (ground_truth_column in df.columns)
    if use_correctness:
        ragas_cols["ground_truth"] = df[ground_truth_column].astype(str).fillna("").tolist()

    ds = Dataset.from_dict(ragas_cols)

    # 5. Khởi tạo LLM & Embeddings (OpenAI hoặc Ollama)
    if os.environ["KEY_API_OPENAI"] != "NULL":
        llm = ChatOpenAI(
            model="gpt-3.5-turbo",
            api_key=os.environ["KEY_API_OPENAI"],
        )
        embeddings = OpenAIEmbeddings(
            model="text-embedding-3-small",
            api_key=os.environ["KEY_API_OPENAI"],
        )
    else:
        llm = ChatOpenAI(
            base_url=os.environ["URL_OLLAMA"],
            model=os.environ["MODEL_CHAT_OLLAMA"],
            api_key=os.environ["API_KEY_OLLAMA"],
        )
        embeddings = OpenAIEmbeddings(
            api_key=os.environ["API_KEY_OLLAMA"],
            base_url=os.environ["URL_OLLAMA"],
            model=os.environ["MODEL_EMBEDDINGS_OLLAMA"],
        )

    # 6. Chọn metrics
    metrics = [context_precision, answer_relevancy, context_recall, faithfulness]
    if use_correctness:
        metrics.append(_answer_correctness)

    # 7. Cấu hình Ragas RunConfig
    run_cfg = RunConfig(
        max_workers=ragas_max_workers,
        timeout=ragas_timeout,
        max_retries=ragas_max_retries,
    )

    # 8. Chạy đánh giá
    try:
        result = evaluate(
            dataset=ds,
            metrics=metrics,
            llm=llm,
            embeddings=embeddings,
            batch_size=ragas_batch_size,
            run_config=run_cfg,
            raise_exceptions=False,  # tránh crash toàn bộ khi 1 batch lỗi
            show_progress=True,
        )
        scored = result.to_pandas()
        # Nối kết quả scores vào DataFrame gốc
        out_df = pd.concat([df.reset_index(drop=True), scored.reset_index(drop=True)], axis=1)
    except Exception as e:
        # Nếu evaluate lỗi, tạo các cột NaN để không mất dữ liệu gốc
        out_df = df.copy()
        for m in metrics:
            col = getattr(m, "name", None) or str(m)
            out_df[col] = float("nan")
        out_df["__error"] = str(e)

    # 9. Xuất Excel
    with pd.ExcelWriter(output_xlsx, engine="openpyxl") as w:
        out_df.to_excel(w, index=False, sheet_name="ragas_scores")

    return os.path.abspath(output_xlsx)
