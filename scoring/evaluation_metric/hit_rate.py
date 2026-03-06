import os
import ast
import pandas as pd


def hit_rate(ground_truth, retrieved_list, k=5):
    """
    Tính Hit Rate cho một truy vấn duy nhất.

    Args:
        ground_truth (str): ground_truth (câu tham chiếu đúng).
        retrieved_list (list[str]): Danh sách context mà hệ thống trả về.
        k (int): Chỉ xét top-k kết quả đầu.

    Returns:
        int:
            - 1 nếu ground_truth xuất hiện trong top-k retrieved_list.
            - 0 nếu không có.
    """
    ground_truth = str(ground_truth).strip()
    top_k = retrieved_list[:k]  # chỉ xét top-k

    for ctx in top_k:
        if ground_truth and ground_truth in str(ctx):
            return 1
    return 0


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


def hit_rate_excel(file_path, out: str = None, k: int = 5):
    """
    Đọc file Excel, tính Hit Rate@k cho từng dòng,
    sau đó tính trung bình toàn bộ và lưu kết quả ra file mới.

    Args:
        file_path (str): Đường dẫn file Excel input.
                         Yêu cầu phải có 2 cột:
                           - "ground_truth": ground_truth
                           - "contexts_answer": danh sách context (string dạng list hoặc list)
        out (str, optional): Đường dẫn file Excel output.
                             Nếu không truyền → mặc định thêm prefix "hit@k_" vào tên gốc.
        k (int): Số lượng top-k context để xét (mặc định 5).

    Returns:
        tuple:
            - str: Đường dẫn file Excel đã tạo.
            - float: Hit Rate trung bình.
    """
    # === Đọc dữ liệu từ Excel ===
    df = pd.read_excel(file_path)

    # === Chuyển cột contexts_ground_truth từ string → list ===
    df["contexts_ground_truth"] = df["contexts_ground_truth"].apply(safe_eval)

    # === Tính Hit@k cho từng dòng ===
    df[f"Hit@{k}"] = df.apply(lambda row: hit_rate(str(row["ground_truth"]), row["contexts_ground_truth"], k), axis=1)

    # === Tính Hit Rate trung bình toàn cục ===
    hit_value = df[f"Hit@{k}"].mean()
    print(f"✅ HitRate@{k} = {hit_value:.4f}")

    # === Tạo đường dẫn output ===
    if not out:
        dir_name = os.path.dirname(file_path)
        base_name = os.path.basename(file_path)
        out = os.path.join(dir_name, f"hit@{k}_" + base_name)

    # === Ghi file mới ra Excel ===
    df.to_excel(out, index=False)
    print(f"✅ Đã tạo file: {out}")

    return out, hit_value
