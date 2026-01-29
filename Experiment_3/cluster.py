import json
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.cluster import AgglomerativeClustering
from collections import defaultdict

# ==========================================
# Configuration
# ==========================================
INPUT_FILE = "slurp_metadata.json"
MODEL_NAME = 'all-MiniLM-L6-v2'  # 高速で標準的な埋め込みモデル
# しきい値が小さいほど厳密に、大きいほど大まかにグループ化されます
DISTANCE_THRESHOLD = 1.2  # 初期クラスタリングの距離しきい値
SIMILARITY_THRESHOLD = 0.6  # グループ所属判定のコサイン類似度しきい値

def cluster_labels(labels, model, distance_threshold=1.2, similarity_threshold=0.6):
    """
    Soft clustering: labels can belong to multiple groups based on similarity to centroids.
    
    Args:
        labels: List of label strings
        model: SentenceTransformer model
        distance_threshold: Threshold for initial clustering to determine centroids
        similarity_threshold: Cosine similarity threshold for group membership
    """
    if not labels:
        return {}

    # 1. 埋め込みベクトルの生成
    embeddings = model.encode(labels)
    
    # 正規化（cosine類似度計算のため）
    normalized_embeddings = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)

    # 2. 初期クラスタリングで中心点を決定
    clustering_model = AgglomerativeClustering(
        n_clusters=None, 
        distance_threshold=distance_threshold, 
        linkage='ward'
    )
    clustering_model.fit(normalized_embeddings)
    
    # 3. 各クラスタの中心点（centroid）を計算
    n_clusters = len(set(clustering_model.labels_))
    centroids = []
    for i in range(n_clusters):
        cluster_mask = clustering_model.labels_ == i
        cluster_points = normalized_embeddings[cluster_mask]
        centroid = cluster_points.mean(axis=0)
        # 中心点も正規化
        centroid = centroid / np.linalg.norm(centroid)
        centroids.append(centroid)
    
    centroids = np.array(centroids)
    
    # 4. 各ラベルについて、すべての中心点との類似度を計算し、所属グループを決定
    clusters = defaultdict(list)
    for idx, (label, vec) in enumerate(zip(labels, normalized_embeddings)):
        # コサイン類似度を計算（正規化済みなので内積で計算可能）
        similarities = np.dot(centroids, vec)
        
        # しきい値を超えるグループをすべて所属先とする
        assigned_groups = [i for i, sim in enumerate(similarities) if sim >= similarity_threshold]
        
        # 所属グループに追加
        for group_id in assigned_groups:
            clusters[group_id].append(label)
    
    return clusters

def find_confusing_pairs(labels, model, threshold=0.8):
    """
    Find pairs of labels that are semantically similar (confusing).
    
    Args:
        labels: List of label strings
        model: SentenceTransformer model
        threshold: Cosine similarity threshold for confusing pairs
    
    Returns:
        List of tuples (label1, label2, similarity_score) sorted by score
    """
    from sentence_transformers import util
    
    if not labels:
        return []
    
    embeddings = model.encode(labels, convert_to_tensor=True)
    # 全ペアのコサイン類似度を計算
    cosine_scores = util.cos_sim(embeddings, embeddings)
    
    confusing_pairs = []
    for i in range(len(labels)):
        for j in range(i + 1, len(labels)):
            score = cosine_scores[i][j].item()
            if score >= threshold:
                confusing_pairs.append((labels[i], labels[j], score))
    
    # 類似度順にソート
    return sorted(confusing_pairs, key=lambda x: x[2], reverse=True)

def print_clusters(title, clusters):
    print(f"\n{'='*20} {title} Clusters {'='*20}")
    for cid, members in sorted(clusters.items()):
        print(f"Group {cid:2}: {', '.join(members)}")

def main():
    # データの読み込み
    try:
        with open(INPUT_FILE, 'r', encoding='utf-8') as f:
            metadata = json.load(f)
    except FileNotFoundError:
        print(f"Error: {INPUT_FILE} not found. Please run the extraction script first.")
        return

    # モデルのロード
    print(f"Loading model '{MODEL_NAME}'...")
    model = SentenceTransformer(MODEL_NAME)

    # 各項目をクラスタリング
    categories = {
        "Scenarios": metadata.get("scenarios", []),
        "Actions": metadata.get("actions", []),
        "Slot Types": metadata.get("slot_types", [])
    }

    results = {}
    confusing_pairs_results = {}
    
    for name, labels in categories.items():
        print(f"\nClustering {name}...")
        clusters = cluster_labels(
            labels, 
            model, 
            distance_threshold=DISTANCE_THRESHOLD,
            similarity_threshold=SIMILARITY_THRESHOLD
        )
        print_clusters(name, clusters)
        results[name.lower()] = clusters
        
        # Confusing pairsの検出
        print(f"\nFinding confusing pairs for {name}...")
        confusing_pairs = find_confusing_pairs(labels, model, threshold=0.8)
        confusing_pairs_results[name.lower()] = [
            {"pair": [p[0], p[1]], "similarity": float(p[2])} 
            for p in confusing_pairs
        ]
        
        if confusing_pairs:
            print(f"Found {len(confusing_pairs)} confusing pairs (similarity >= 0.8):")
            for label1, label2, score in confusing_pairs[:10]:  # 上位10件表示
                print(f"  - {label1} <-> {label2}: {score:.4f}")
            if len(confusing_pairs) > 10:
                print(f"  ... and {len(confusing_pairs) - 10} more pairs")
        else:
            print("  No confusing pairs found.")

    # クラスタリング結果をJSONで保存
    output_file = "slurp_clusters.json"
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=4, ensure_ascii=False)
    print(f"\n✓ Clustering results saved to {output_file}")
    
    # Confusing pairs結果をJSONで保存
    confusing_output_file = "slurp_confusing_pairs.json"
    with open(confusing_output_file, 'w', encoding='utf-8') as f:
        json.dump(confusing_pairs_results, f, indent=4, ensure_ascii=False)
    print(f"✓ Confusing pairs results saved to {confusing_output_file}")

if __name__ == "__main__":
    main()