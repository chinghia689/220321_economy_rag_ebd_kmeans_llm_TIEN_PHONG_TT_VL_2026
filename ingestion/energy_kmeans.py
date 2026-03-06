import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import warnings
warnings.filterwarnings("ignore", category=UserWarning) # Ẩn cảnh báo của K-Means
from sklearn.metrics.pairwise import cosine_similarity
from ingestion.energy_base_distance import energy_base_distance


class EnergyRetriever:
    """
    Module Truy xuất thông tin nâng cao sử dụng Energy-Based Distance và K-Means.
    """
    def __init__(self, vector_store, embeddings_model, k_retrieve=40, similarity_threshold=0.50, n_top_clusters=1, max_final_docs=15):
        """
        Khởi tạo Energy Retriever.
        
        Args:
            vector_store: Chroma vector store
            embeddings_model: Model embedding (HuggingFace embeddings)
            k_retrieve: Số top documents để retrieve (mặc định 40)
            similarity_threshold: Ngưỡng cosine similarity (mặc định 0.50)
            n_top_clusters: Số clusters tốt nhất để lấy docs (mặc định 1)
            max_final_docs: Số documents tối đa trả về cuối cùng (mặc định 15)
        """
        # retriever chuẩn dùng Cosine (Lấy diện rộng)
        self.retriever = vector_store.as_retriever(search_kwargs={'k': k_retrieve})
        self.embeddings = embeddings_model
        self.similarity_threshold = similarity_threshold
        self.n_top_clusters = n_top_clusters
        self.max_final_docs = max_final_docs
        self.vector_store = vector_store

    def retrieve(self, query):
        """
        Truy xuất documents dựa trên query sử dụng Energy Distance.
        
        Args:
            query (str): Câu hỏi/query của người dùng
            
        Returns:
            List[Document]: Danh sách Document objects liên quan nhất
        """
        print(f"\n🔎 [Energy Retriever] Đang xử lý câu hỏi: '{query}'")
        
        # 1. Truy xuất diện rộng (Top 40 từ cosine similarity)
        docs = self.retriever.invoke(query)
        if not docs:
            print("   -> ⚠️ Không tìm thấy tài liệu thô nào.")
            return []

        context = [doc.page_content for doc in docs]

        # 2. Embedding lại query và context
        # (Cách này tốn kém vì phải embed lại, nhưng an toàn và dễ code)
        doc_vectors = np.array(self.embeddings.embed_documents(context))
        query_vector = np.array(self.embeddings.embed_query(query)).reshape(1, -1)

        # 3. Tính cosine similarity cho từng doc
        sims = cosine_similarity(query_vector, doc_vectors)[0]
        max_sim = np.max(sims)
        print(f"   -> Max Cosine Similarity: {max_sim:.4f}")
        
        if max_sim < self.similarity_threshold:
            print(f"   -> 🛑 Dữ liệu nhiễu (Dưới ngưỡng {self.similarity_threshold}). Ngắt luồng!")
            return []

        # 4. Lọc: chỉ giữ docs có cosine >= threshold
        valid_mask = sims >= self.similarity_threshold
        valid_indices = np.where(valid_mask)[0]
        filtered_vectors = doc_vectors[valid_indices]
        filtered_sims = sims[valid_indices]
        print(f"   -> 📋 Lọc cosine >= {self.similarity_threshold}: giữ {len(valid_indices)}/{len(docs)} docs")

        # 5. Nếu quá ít docs, trả về trực tiếp (không cần clustering)
        if len(valid_indices) <= 3:
            sorted_idx = np.argsort(-filtered_sims)  # Sắp xếp cosine giảm dần
            final_docs = [docs[valid_indices[i]] for i in sorted_idx]
            print(f"   -> ✅ Ít docs, trả trực tiếp {len(final_docs)} documents (đã sắp xếp theo cosine)")
            return final_docs

        # 6. Gom cụm K-Means trên docs đã lọc
        n_samples = len(filtered_vectors)
        
        # Đặt giới hạn K chạy thử: Ít nhất là 2, nhiều nhất là 10 (hoặc nhỏ hơn nếu số lượng docs ít)
        max_possible_k = min(10, n_samples - 1)
        
        best_k = 2  # Khởi tạo mặc định
        best_labels = None
        
        if n_samples > 2: # Chỉ gom cụm khi có từ 3 docs trở lên
            best_score = -1.0
            
            for k in range(2, max_possible_k + 1):
                kmeans_temp = KMeans(n_clusters=k, random_state=42, n_init='auto')
                labels_temp = kmeans_temp.fit_predict(filtered_vectors)
                
                # Tính điểm Silhouette cho cách chia K này
                score = silhouette_score(filtered_vectors, labels_temp)
                
                if score > best_score:
                    best_score = score
                    best_k = k
                    best_labels = labels_temp
            
            print(f"   -> 🤖 Tự động chọn K tối ưu = {best_k} (Silhouette Score cao nhất: {best_score:.4f})")
            labels = best_labels
            actual_k = best_k
            
        else:
            # Nếu chỉ có 2 docs, ép thành 1 cụm hoặc chia làm 2 tùy bạn (ở đây chia 1 cho an toàn)
            best_k = 1
            labels = np.zeros(n_samples, dtype=int)
            actual_k = best_k
            print(f"   -> ⚠️ Số lượng docs quá ít ({n_samples}), tự động gom thành 1 cụm.")


        # 7. Tính Energy Distance cho từng cụm và xếp hạng
        cluster_energies = []
        for i in range(actual_k):
            cluster_mask = labels == i
            if not np.any(cluster_mask):
                continue
                
            cluster_vectors = filtered_vectors[cluster_mask]
            energy = energy_base_distance(query_vector, cluster_vectors)
            cluster_energies.append((i, energy))
        
        # Sắp xếp theo energy distance tăng dần (thấp = tốt)
        cluster_energies.sort(key=lambda x: x[1])
        
        # 8. Lấy docs từ top N clusters
        n_select = min(self.n_top_clusters, len(cluster_energies))
        selected_clusters = cluster_energies[:n_select]
        
        for idx, (cluster_id, energy) in enumerate(selected_clusters):
            print(f"   -> {'🏆' if idx == 0 else '📌'} Cụm {cluster_id} - Energy Distance = {energy:.4f}")
        
        # Gom docs từ các cụm được chọn
        all_selected_indices = []
        all_selected_sims = []
        for cluster_id, _ in selected_clusters:
            win_mask = labels == cluster_id
            win_local_indices = np.where(win_mask)[0]
            all_selected_indices.extend(win_local_indices)
            all_selected_sims.extend(filtered_sims[win_local_indices])
        
        all_selected_indices = np.array(all_selected_indices)
        all_selected_sims = np.array(all_selected_sims)
        
        # Re-rank tất cả docs theo cosine similarity (cao → thấp)
        sorted_order = np.argsort(-all_selected_sims)
        
        # Loại bỏ trùng lặp (giữ thứ tự cosine cao nhất)
        seen = set()
        final_docs = []
        for i in sorted_order:
            original_idx = valid_indices[all_selected_indices[i]]
            if original_idx not in seen:
                seen.add(original_idx)
                final_docs.append(docs[original_idx])
                if len(final_docs) >= self.max_final_docs:
                    break

        print(f"   -> ✅ Truy xuất {len(final_docs)} documents từ top {n_select} clusters (max {self.max_final_docs}, re-ranked by cosine)")
        return final_docs

