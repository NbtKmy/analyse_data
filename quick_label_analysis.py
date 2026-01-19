"""
ラベル分布の簡易分析スクリプト
"""

import pandas as pd
import numpy as np
from pathlib import Path
import ast
from collections import Counter

# データ読み込み
DATA_DIR = Path("dev_data")
csv_files = sorted(DATA_DIR.glob("rvk_chunk_*.csv"))

dfs = []
for csv_file in csv_files:
    df_chunk = pd.read_csv(csv_file)
    dfs.append(df_chunk)

df = pd.concat(dfs, ignore_index=True)

# RVK表記のパース
def parse_rvk_notations(notation_str):
    try:
        if pd.isna(notation_str) or notation_str == '[]':
            return []
        notations = ast.literal_eval(notation_str)
        return [notation[0] if isinstance(notation, tuple) else notation for notation in notations]
    except:
        return []

df['rvk_labels'] = df['unique_rvk_notations'].apply(parse_rvk_notations)
df['num_labels'] = df['rvk_labels'].apply(len)

# ラベルありのデータのみ
df_with_labels = df[df['num_labels'] > 0].copy()

# 全ラベルを抽出
all_labels = []
for labels in df_with_labels['rvk_labels']:
    all_labels.extend(labels)

label_counter = Counter(all_labels)

print("=" * 80)
print("RVKラベル分布の詳細分析")
print("=" * 80)

print(f"\n基本統計:")
print(f"  総ラベル出現数: {len(all_labels):,}")
print(f"  ユニークラベル数: {len(label_counter):,}")
print(f"  平均出現回数: {len(all_labels) / len(label_counter):.2f}")

# ラベルごとのサンプル数分布
label_counts = pd.Series(label_counter).sort_values(ascending=False)

print(f"\nラベルごとのサンプル数統計:")
print(f"  最大: {label_counts.max():,}回")
print(f"  75%点: {int(label_counts.quantile(0.75)):,}回")
print(f"  中央値: {int(label_counts.median()):,}回")
print(f"  25%点: {int(label_counts.quantile(0.25)):,}回")
print(f"  最小: {label_counts.min():,}回")
print(f"  平均: {label_counts.mean():.2f}回")

print(f"\n🚨 重大な問題: 中央値が{int(label_counts.median())}回 = 半数のラベルが{int(label_counts.median())}回以下")

# サンプル数ごとのラベル数
print(f"\nLong-tail分析:")
print(f"  サンプル数が1回のみ: {(label_counts == 1).sum():,}個 ({(label_counts == 1).sum()/len(label_counts)*100:.1f}%)")
print(f"  サンプル数が2-4回: {((label_counts >= 2) & (label_counts <= 4)).sum():,}個 ({((label_counts >= 2) & (label_counts <= 4)).sum()/len(label_counts)*100:.1f}%)")
print(f"  サンプル数が5-9回: {((label_counts >= 5) & (label_counts <= 9)).sum():,}個 ({((label_counts >= 5) & (label_counts <= 9)).sum()/len(label_counts)*100:.1f}%)")
print(f"  サンプル数が10-49回: {((label_counts >= 10) & (label_counts < 50)).sum():,}個 ({((label_counts >= 10) & (label_counts < 50)).sum()/len(label_counts)*100:.1f}%)")
print(f"  サンプル数が50-99回: {((label_counts >= 50) & (label_counts < 100)).sum():,}個 ({((label_counts >= 50) & (label_counts < 100)).sum()/len(label_counts)*100:.1f}%)")
print(f"  サンプル数が100回以上: {(label_counts >= 100).sum():,}個 ({(label_counts >= 100).sum()/len(label_counts)*100:.1f}%)")

# 実用的なラベル数の評価
print(f"\n実用的なラベル数（最低サンプル数でフィルタリング）:")
for threshold in [5, 10, 20, 50, 100]:
    viable = (label_counts >= threshold).sum()
    print(f"  最低{threshold:3d}サンプル以上: {viable:,}ラベル ({viable/len(label_counts)*100:.1f}%)")

# RVK階層構造の分析
def extract_main_category(rvk_code):
    """RVKコードから主分類を抽出"""
    if not isinstance(rvk_code, str):
        return None
    parts = rvk_code.strip().split()
    if parts:
        main = ''.join([c for c in parts[0] if c.isalpha()])
        return main if main else None
    return None

main_categories = {}
for label, count in label_counter.items():
    main_cat = extract_main_category(label)
    if main_cat:
        if main_cat not in main_categories:
            main_categories[main_cat] = {'count': 0, 'labels': 0}
        main_categories[main_cat]['count'] += count
        main_categories[main_cat]['labels'] += 1

print(f"\nRVK主分類の統計:")
print(f"  主分類の数: {len(main_categories)}")
sorted_main = sorted(main_categories.items(), key=lambda x: x[1]['count'], reverse=True)
print(f"\n  上位10主分類:")
for main_cat, stats in sorted_main[:10]:
    print(f"    {main_cat:5s}: {stats['count']:6,}出現, {stats['labels']:5,}サブラベル")

# フィルタリングシミュレーション
print(f"\n" + "=" * 80)
print("フィルタリングシミュレーション（最低50サンプル）")
print("=" * 80)

min_samples = 50
viable_label_set = set(label_counts[label_counts >= min_samples].index)

def has_viable_label(labels):
    return any(label in viable_label_set for label in labels)

df_filtered = df_with_labels[df_with_labels['rvk_labels'].apply(has_viable_label)].copy()

def filter_labels(labels):
    return [label for label in labels if label in viable_label_set]

df_filtered['filtered_labels'] = df_filtered['rvk_labels'].apply(filter_labels)
df_filtered['num_filtered_labels'] = df_filtered['filtered_labels'].apply(len)

print(f"\n結果:")
print(f"  残るラベル数: {len(viable_label_set):,} / 27,051 ({len(viable_label_set)/27051*100:.1f}%)")
print(f"  残るレコード数: {len(df_filtered):,} / {len(df_with_labels):,} ({len(df_filtered)/len(df_with_labels)*100:.1f}%)")
print(f"  平均ラベル数/レコード: {df_filtered['num_filtered_labels'].mean():.2f}")

# フィルタリング後のラベル分布
filtered_label_counter = Counter()
for labels in df_filtered['filtered_labels']:
    filtered_label_counter.update(labels)

filtered_label_counts = pd.Series(filtered_label_counter).sort_values(ascending=False)
print(f"\nフィルタリング後のラベル統計:")
print(f"  最小サンプル数: {filtered_label_counts.min():,}回")
print(f"  最大サンプル数: {filtered_label_counts.max():,}回")
print(f"  平均サンプル数: {filtered_label_counts.mean():.2f}回")
print(f"  中央値: {int(filtered_label_counts.median()):,}回")

print(f"\n" + "=" * 80)
print("推奨事項")
print("=" * 80)

print(f"""
🔴 現状のまま全27,051ラベルで学習: 困難
   理由: データ不足ラベルが多すぎる

🟢 推奨アプローチ:
   1. 最低50-100サンプルでフィルタリング → {len(viable_label_set):,}ラベル
   2. 階層的分類（主分類{len(main_categories)}個 → 詳細分類）
   3. Few-shot学習の技術を併用

💡 次のステップ:
   1. フィルタリング閾値を決定（50を推奨）
   2. 小規模な実験から開始
   3. 評価指標の定義（Precision@K, Recall@Kなど）
""")
