# Experiment_2: Confusable Pairs Analysis

Speech LLM (Qwen2-Audio) のファインチューニングにおける「混同ラベル（Confusable Pairs）」を分析。

## 使い方

### 基本実行
```bash
./Experiment_2/run_analysis.sh outputs/audio_text_mix
```

### キャッシュを使って再分析（推論をスキップ）
```bash
./Experiment_2/run_analysis.sh outputs/audio_text_mix Experiment_2/output --use_cache
```

### 個別実行
```bash
# 1. 推論 + 特徴抽出（最初に1回だけ実行）
python Experiment_2/run_analysis.py \
    --model_path outputs/audio_text_mix \
    --output_dir Experiment_2/output \
    --max_samples 100  # テスト用

# 2. 可視化（キャッシュされたデータを使用）
python Experiment_2/visualize_features.py --input_dir Experiment_2/output
python Experiment_2/entropy_analysis.py --input_dir Experiment_2/output

# 3. 再分析（推論なし）
python Experiment_2/run_analysis.py \
    --model_path outputs/audio_text_mix \
    --output_dir Experiment_2/output \
    --use_cache
```

## 出力ファイル

| ファイル | 説明 |
|---------|------|
| `cached_inference_data.pt` | 全特徴量キャッシュ（再分析用） |
| `hidden_states.pt` | 隠れ状態ベクトル |
| `logits.pt` | 出力ロジット |
| `confusion_matrix.npy` | 混同行列 |
| `analysis_summary.json` | 分析サマリー |
| `sample_results.json` | サンプル別結果 |
| `figures/` | t-SNE/UMAP可視化 |
| `entropy_figures/` | エントロピー分析 |

## スクリプト

| ファイル | 説明 |
|---------|------|
| `run_analysis.py` | メイン分析（推論+特徴抽出） |
| `visualize_features.py` | t-SNE/UMAP可視化 |
| `attention_analysis.py` | アテンション解析 |
| `entropy_analysis.py` | エントロピー分析 |

## サンプル分類

- `success`: 高確信度で正解
- `ambiguous_success`: 低確信度だが正解
- `ambiguous_failure`: 近接ラベルに誤分類
- `fatal_failure`: 全く別のラベルを予測

## 依存パッケージ

```bash
pip install scikit-learn matplotlib seaborn umap-learn
```
