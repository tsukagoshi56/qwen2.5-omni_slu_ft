import matplotlib.pyplot as plt
import numpy as np

# ==========================================
# 0. Configuration & Dummy Data
# ==========================================
# (正解GT, 予測Pred, 間違い回数)
dummy_errors = [
    # --- 1. ブラックホール現象 (The Black Hole) ---
    ("weather", "general", 150),
    ("news",    "general", 120),
    ("alarm",   "general", 80),
    ("music",   "general", 90),
    
    # --- 2. 階層的混同 (Hierarchical Confusion) ---
    ("joke",    "quirky",  60),
    ("satire",  "quirky",  40),

    # --- 3. 近傍誤認 (Neighbor Confusion) ---
    ("play",    "pause",   30),
    ("pause",   "play",    20),
    ("alarm",   "timer",   25),
    
    # --- 4. 複合要因 ---
    ("play",    "general", 50),
]

# 座標定義 (意味空間の配置を模倣)
coordinates = {
    "general": (0, 0),       # Center Black Hole
    "quirky":  (3, 3),       # Abstract Zone
    "joke":    (3.5, 4.0),
    "satire":  (2.8, 3.8),
    "music":   (-3, -3),     # Media Zone
    "play":    (-3.5, -3.2),
    "pause":   (-3.2, -3.5),
    "weather": (-3, 3),      # Info Zone
    "news":    (-2.5, 3.5),
    "alarm":   (3, -3),      # Time Zone
    "timer":   (3.5, -2.8),
}

# ==========================================
# 1. Visualization Logic
# ==========================================
def draw_gravity_field():
    plt.figure(figsize=(12, 10), facecolor='white')
    ax = plt.gca()
    
    # 集計
    pred_counts = {}
    for gt, pred, count in dummy_errors:
        pred_counts[pred] = pred_counts.get(pred, 0) + count
        if gt not in pred_counts: pred_counts[gt] = 0

    max_count = max(pred_counts.values()) if pred_counts else 1

    # --- A. ノード描画 ---
    for label, (x, y) in coordinates.items():
        count = pred_counts.get(label, 0)
        
        # サイズ計算 (強調のため大きく)
        base_size = 300
        size_factor = (count / max_count) * 5000
        size = base_size + size_factor
        
        # 色分け: ブラックホールは赤、被害者は青
        is_predator = count > max_count * 0.2
        color = '#e74c3c' if is_predator else '#3498db'
        font_weight = 'bold' if is_predator else 'normal'
        font_size = 14 if is_predator else 10
        zorder = 10 if is_predator else 5
        
        # 円を描画
        plt.scatter(x, y, s=size, c=color, alpha=0.7, edgecolors='white', linewidth=1.5, zorder=zorder)
        
        # ラベル
        plt.text(x, y, label, ha='center', va='center', 
                 fontsize=font_size, fontweight=font_weight, color='white', zorder=zorder+1)

    # --- B. 矢印（重力フロー）描画 ---
    for gt, pred, count in dummy_errors:
        if gt in coordinates and pred in coordinates:
            start = coordinates[gt]
            end = coordinates[pred]
            
            # 太さと濃さ
            width = 1.0 + (count / max_count) * 6.0
            alpha = 0.4 + (count / max_count) * 0.6
            
            ax.annotate("",
                        xy=end, xycoords='data',
                        xytext=start, textcoords='data',
                        arrowprops=dict(arrowstyle="->",
                                        connectionstyle="arc3,rad=0.1", # カーブ
                                        color="#c0392b",
                                        lw=width,
                                        alpha=alpha),
                        zorder=4)

    # 装飾
    plt.title("Semantic Gravity Field Analysis\n(Red Circles = 'Black Hole' Labels absorbing specific intents)", fontsize=16, pad=20)
    plt.xlabel("Semantic Dimension 1")
    plt.ylabel("Semantic Dimension 2")
    plt.grid(True, linestyle='--', alpha=0.3)
    
    # 軸の数値を消す（概念図っぽくする）
    plt.xticks([])
    plt.yticks([])
    
    plt.tight_layout()
    # 保存と表示の両方を試みる（headless環境用）
    save_path = "Experiment_2/gravity_field.png"
    plt.savefig(save_path)
    print(f"Visualization saved to {save_path}")
    # plt.show() # headlessではエラーになるためコメントアウト

draw_gravity_field()