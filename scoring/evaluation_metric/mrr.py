import os
import ast
import pandas as pd


def reciprocal_rank(reference, retrieved_list):
    """
    Tính Reciprocal Rank (RR) cho một truy vấn duy nhất.

    Args:
        reference (str): Ground truth (câu đúng cần tìm).
        retrieved_list (list[str]): Danh sách context do hệ thống trả về
                                    (theo thứ tự từ cao xuống thấp).

    Returns:
        float:
            - Reciprocal Rank = 1 / rank (rank là vị trí đầu tiên mà reference xuất hiện).
            - Nếu không tìm thấy → 0.0.
    """
    reference = str(reference).strip()

    # Duyệt từng phần tử trong retrieved_list theo thứ tự
    for idx, ctx in enumerate(retrieved_list, start=1):
        if reference and reference in str(ctx):
            return 1.0 / idx  # trả về nghịch đảo vị trí
    return 0.0


def safe_eval(x):
    """
    Parse string thành list an toàn bằng ast.literal_eval.

    Args:
        x (str | list): Chuỗi dạng list (ví dụ "['a','b']") hoặc list.

    Returns:
        list: List đã parse thành công, hoặc [] nếu lỗi.
    """
    if isinstance(x, str):
        try:
            return ast.literal_eval(x)
        except Exception:
            return []
    elif isinstance(x, list):
        return x
    return []


def mrr_excel(file_path, out: str = None):
    """
    Đọc file Excel, tính Reciprocal Rank cho từng dòng,
    sau đó tính MRR (Mean Reciprocal Rank) toàn cục.

    Args:
        file_path (str): Đường dẫn file Excel input.
                         Yêu cầu có cột:
                           - "reference": ground truth
                           - "contexts_answer": danh sách context (string dạng list hoặc list).
        out (str, optional): Đường dẫn file Excel output.
                             Nếu không truyền → mặc định thêm prefix "mrr_" vào tên gốc.

    Returns:
        tuple:
            - str: Đường dẫn file Excel đã tạo.
            - float: Mean Reciprocal Rank (MRR) toàn cục.
    """
    # === Đọc dữ liệu từ Excel ===
    df = pd.read_excel(file_path)

    # === Parse contexts_ground_truth từ string → list ===
    df["contexts_ground_truth"] = df["contexts_ground_truth"].apply(safe_eval)

    # === Tính Reciprocal Rank cho từng dòng ===
    df["Mean Reciprocal Rank"] = df.apply(lambda row: reciprocal_rank(str(row["ground_truth"]), row["contexts_ground_truth"]), axis=1)

    # === Tính MRR toàn cục ===
    mrr_value = df["Mean Reciprocal Rank"].mean()
    print(f"✅ MRR = {mrr_value:.4f}")

    # === Tạo đường dẫn output ===
    if not out:
        dir_name = os.path.dirname(file_path)
        base_name = os.path.basename(file_path)
        out = os.path.join(dir_name, "mrr_" + base_name)

    # === Ghi ra Excel mới ===
    df.to_excel(out, index=False)
    print(f"✅ Đã tạo file: {out}")

    return out, mrr_value
