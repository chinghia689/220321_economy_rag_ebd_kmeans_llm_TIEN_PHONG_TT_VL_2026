"""
Script tạo file evaluation từ chatbot.

Bước 1: Chạy script này để chatbot trả lời các câu hỏi
Bước 2: Mở file Excel, điền cột "ground_truth" thủ công
Bước 3: Chạy scoring/main.py để chấm điểm
"""

import os
import sys
import glob
from pathlib import Path

# Thêm parent folder vào path
sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd
from chatbot.main import ChatbotRunner


# ============================================================
# HÀM LOAD 1000 CÂU HỎI TỪ FILE EXCEL
# ============================================================

def _build_filename_index(dataset_dir: str) -> dict:
    """
    Duyệt toàn bộ Dataset_economy/ và tạo dict {filename: filepath}.
    Nếu 1 filename xuất hiện ở nhiều thư mục → lấy file đầu tiên tìm được.
    """
    index = {}
    for fpath in glob.glob(os.path.join(dataset_dir, "**", "*.txt"), recursive=True):
        fname = os.path.basename(fpath)
        if fname not in index:
            index[fname] = fpath
    return index


def load_questions_from_excel(
    excel_path: str,
    dataset_dir: str,
    max_questions: int = None,
) -> list:
    """
    Đọc file Excel 1000 câu hỏi và resolve contexts_ground_truth
    từ tên file .txt thành nội dung text đầy đủ.

    Args:
        excel_path:  Đường dẫn file Excel (vd: scoring/file 1000 cau hoi.xlsx)
        dataset_dir: Thư mục chứa các file .txt (vd: Dataset_economy/)
        max_questions: Giới hạn số câu hỏi (None = lấy tất cả)

    Returns:
        list[dict]: Danh sách dict với keys: question, ground_truth, contexts_ground_truth
    """
    df = pd.read_excel(excel_path)
    if max_questions:
        df = df.head(max_questions)

    print(f"📂 Đang xây dựng index file .txt từ {dataset_dir}...")
    file_index = _build_filename_index(dataset_dir)
    print(f"   → Tìm thấy {len(file_index)} file .txt duy nhất")

    results = []
    not_found = 0

    for _, row in df.iterrows():
        question = str(row["question"]).strip()
        ground_truth = str(row["ground_truth"]).strip()
        ctx_filename = str(row["contexts_ground_truth"]).strip()

        # Resolve filename → nội dung text
        contexts_text = ""
        if ctx_filename in file_index:
            try:
                with open(file_index[ctx_filename], "r", encoding="utf-8") as f:
                    contexts_text = f.read().strip()
            except Exception as e:
                print(f"   ⚠️ Lỗi đọc file {ctx_filename}: {e}")
        else:
            not_found += 1

        results.append({
            "question": question,
            "ground_truth": ground_truth,
            "contexts_ground_truth": contexts_text,
        })

    if not_found:
        print(f"   ⚠️ {not_found}/{len(df)} file .txt không tìm thấy trong {dataset_dir}")
    print(f"✅ Đã load {len(results)} câu hỏi từ {excel_path}")

    return results


# ============================================================
# HÀM TẠO FILE EVALUATION
# ============================================================

def create_evaluation_file(questions: list, output_file: str = "eval_data.xlsx", ground_truths: dict = None, contexts_gt: dict = None):
    """
    Chạy chatbot với danh sách câu hỏi và lưu kết quả ra Excel.
    
    Args:
        questions: Danh sách câu hỏi cần test
        output_file: Tên file Excel output
        ground_truths: Dict {question: ground_truth} để tự động điền đáp án
        contexts_gt: Dict {question: contexts_ground_truth} để tự động điền context tham chiếu
    """
    
    # Khởi tạo chatbot
    print("🚀 Đang khởi tạo chatbot...")
    chatbot = ChatbotRunner(
        path_vector_store="./chroma_economy_db",
        llm_provider="openai"
    )
    
    results = []
    
    for i, question in enumerate(questions, 1):
        print(f"\n{'='*60}")
        print(f"📝 [{i}/{len(questions)}] Câu hỏi: {question}")
        print(f"{'='*60}")
        
        # Chuẩn bị input
        input_state = {
            "question": question,
            "generation": "",
            "documents": [],
            "prompt": "Bạn là một chuyên gia tư vấn kinh tế Việt Nam. Hãy trả lời câu hỏi CHỈ dựa trên thông tin trong ngữ cảnh được cung cấp. Nếu ngữ cảnh không chứa thông tin cần thiết, hãy nói rõ là không có thông tin."
        }
        
        # Chạy workflow
        try:
            output_state = chatbot.compiled_workflow.invoke(input_state)
            
            answer = output_state.get("generation", "")
            documents = output_state.get("documents", [])
            
            # Lấy contexts từ documents
            contexts = [doc.page_content for doc in documents]
            
            gt = ground_truths.get(question, "") if ground_truths else ""
            ctx_gt = contexts_gt.get(question, "") if contexts_gt else ""
            results.append({
                "question": question,
                "ground_truth": gt,
                "contexts_ground_truth": str([ctx_gt]) if ctx_gt else "[]",
                "answer": answer,
                "contexts_answer": str(contexts),
                "metadata": str([doc.metadata for doc in documents]) if documents else ""
            })
            
            print(f"✅ Đã xử lý thành công")
            
        except Exception as e:
            print(f"❌ Lỗi: {e}")
            results.append({
                "question": question,
                "ground_truth": "",
                "contexts_ground_truth": "",
                "answer": f"ERROR: {e}",
                "contexts_answer": "[]",
                "metadata": ""
            })
    
    # Lưu ra Excel
    df = pd.DataFrame(results)
    output_path = os.path.join(os.path.dirname(__file__), output_file)
    df.to_excel(output_path, index=False)
    
    print(f"\n{'='*60}")
    print(f"✅ Đã lưu {len(results)} kết quả vào: {output_path}")
    print(f"   Sau đó chạy: python scoring/main.py")
    print(f"{'='*60}")
    
    return output_path


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Tạo file evaluation từ chatbot")
    parser.add_argument("--source", choices=["excel", "hardcode"], default="excel",
                        help="Nguồn câu hỏi: 'excel' (file 1000 câu) hoặc 'hardcode' (30 câu cũ)")
    parser.add_argument("--max", type=int, default=None,
                        help="Giới hạn số câu hỏi (mặc định: tất cả)")
    parser.add_argument("--output", type=str, default="eval_data.xlsx",
                        help="Tên file output (mặc định: eval_data.xlsx)")
    args = parser.parse_args()

    project_root = str(Path(__file__).parent.parent)

    if args.source == "excel":
        # === LOAD TỪ FILE 1000 CÂU HỎI ===
        excel_path = os.path.join(os.path.dirname(__file__), "file 1000 cau hoi.xlsx")
        dataset_dir = os.path.join(project_root, "Dataset_economy")

        if not os.path.exists(excel_path):
            print(f"❌ Không tìm thấy file: {excel_path}")
            sys.exit(1)

        qa_list = load_questions_from_excel(excel_path, dataset_dir, max_questions=args.max)

        questions = [q["question"] for q in qa_list]
        ground_truths = {q["question"]: q["ground_truth"] for q in qa_list}
        contexts_gt = {q["question"]: q["contexts_ground_truth"] for q in qa_list}

        create_evaluation_file(questions, args.output, ground_truths=ground_truths, contexts_gt=contexts_gt)

    else:
        print("❌ Source 'hardcode' không khả dụng. Dùng --source excel")
        sys.exit(1)


