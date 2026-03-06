

import os
import sys
from pathlib import Path

# Thêm parent folder vào path
sys.path.insert(0, str(Path(__file__).parent.parent))

from scoring.evaluation_metric.rouge_n import rouge_excel
from scoring.evaluation_metric.bleu import bleu_excel
from scoring.evaluation_metric.cosine_similarity import cosine_excel
from scoring.evaluation_metric.mrr import mrr_excel
from scoring.evaluation_metric.hit_rate import hit_rate_excel
from scoring.evaluation_metric.ndcg import ndcg_excel
# from scoring.evaluation_metric.ragas import score_excel_with_ragas_to_xlsx


def evaluate_results(file_path, embeddings):
    """
    Chạy toàn bộ pipeline đánh giá: ragas, he, rouge, bleu, cosine,
    MRR, hit rate, NDCG, aggregate.

    Args:
        file_path (str): File Excel chứa kết quả hỏi đáp.
        embeddings: Embedding để tính cosine similarity.

    Returns:
        str: Đường dẫn file kết quả cuối cùng.
    """

    # # RAGAS (faithfulness, answer_relevancy, context_precision, context_recall, answer_correctness)
    # try:
    #     ragas_output = os.path.join(os.path.dirname(file_path), "ragas_scores.xlsx")
    #     ragas_result = score_excel_with_ragas_to_xlsx(file_path, output_xlsx=ragas_output)
    #     print(f"✅ Ragas scores saved to: {ragas_result}")
    # except Exception as e:
    #     print(f"⚠️ Ragas scoring failed (skipping): {e}")

    # ROUGE-N (độ giống nhau dựa trên n-gram)
    file_path = rouge_excel(file_path, n=2)

    # BLEU (so khớp dịch máy)
    file_path = bleu_excel(file_path, n=2)

    # Cosine similarity (đo tương đồng vector)
    file_path = cosine_excel(file_path, embeddings)

    # MRR (Mean Reciprocal Rank)
    file_path, mrr_value = mrr_excel(file_path)

    # Hit Rate@k
    file_path, hit_value = hit_rate_excel(file_path, k=5)

    # NDCG@k (Normalized Discounted Cumulative Gain)
    file_path = ndcg_excel(file_path, k=5)

    print(f"✅ Evaluation done. MRR={mrr_value} | HIT={hit_value}")

    return file_path


if __name__ == "__main__":
    from ingestion.model_embedding import vn_embedder
    
    # Lấy embedding model tiếng Việt
    embeddings = vn_embedder.get_model()
    
    # File evaluation data
    eval_file = os.path.join(os.path.dirname(__file__), "eval_data.xlsx")
    
    if os.path.exists(eval_file):
        result = evaluate_results(eval_file, embeddings)
        print(f"📁 Kết quả lưu tại: {result}")
    else:
        print(f"❌ Không tìm thấy file: {eval_file}")