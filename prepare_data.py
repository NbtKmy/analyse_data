"""
データ前処理スクリプト
出現回数50以上のラベルでフィルタリングし、train/val/testに分割
"""

import pandas as pd
import numpy as np
from pathlib import Path
import ast
import json
from collections import Counter
from sklearn.model_selection import train_test_split

def parse_rvk_notations(notation_str):
    """RVK表記の文字列をパース"""
    try:
        if pd.isna(notation_str) or notation_str == '[]':
            return []
        notations = ast.literal_eval(notation_str)
        return [notation[0] if isinstance(notation, tuple) else notation for notation in notations]
    except:
        return []

def main():
    # データ読み込み
    print("=" * 80)
    print("RVKデータの前処理")
    print("=" * 80)

    DATA_DIR = Path("dev_data")
    csv_files = sorted(DATA_DIR.glob("rvk_chunk_*.csv"))

    print(f"\n発見されたCSVファイル数: {len(csv_files)}")

    dfs = []
    for csv_file in csv_files:
        df_chunk = pd.read_csv(csv_file)
        dfs.append(df_chunk)

    df = pd.concat(dfs, ignore_index=True)
    print(f"総レコード数: {len(df):,}")

    # RVKラベルをパース
    print("\nRVKラベルをパース中...")
    df['rvk_labels'] = df['unique_rvk_notations'].apply(parse_rvk_notations)
    df['num_labels'] = df['rvk_labels'].apply(len)

    # ラベルありのデータのみ
    df_with_labels = df[df['num_labels'] > 0].copy()
    print(f"ラベル付きレコード数: {len(df_with_labels):,}")

    # 全ラベルを抽出してカウント
    print("\nラベル出現回数をカウント中...")
    all_labels = []
    for labels in df_with_labels['rvk_labels']:
        all_labels.extend(labels)

    label_counter = Counter(all_labels)
    print(f"ユニークラベル数: {len(label_counter):,}")

    # 最低50サンプルでフィルタリング
    min_samples = 50
    viable_label_set = set([label for label, count in label_counter.items() if count >= min_samples])

    print(f"\n最低{min_samples}サンプル以上のラベル数: {len(viable_label_set):,}")

    # フィルタリング
    def has_viable_label(labels):
        return any(label in viable_label_set for label in labels)

    def filter_labels(labels):
        return [label for label in labels if label in viable_label_set]

    df_filtered = df_with_labels[df_with_labels['rvk_labels'].apply(has_viable_label)].copy()
    df_filtered['rvk_labels'] = df_filtered['rvk_labels'].apply(filter_labels)
    df_filtered['num_labels'] = df_filtered['rvk_labels'].apply(len)

    print(f"フィルタリング後のレコード数: {len(df_filtered):,}")
    print(f"平均ラベル数/レコード: {df_filtered['num_labels'].mean():.2f}")

    # 入力テキストの作成（タイトル、著者、出版社を結合）
    print("\n入力テキストを作成中...")
    def create_input_text(row):
        parts = []
        if pd.notna(row['consolidated_title']) and row['consolidated_title']:
            parts.append(f"Title: {row['consolidated_title']}")
        if pd.notna(row['author']) and row['author']:
            parts.append(f"Author: {row['author']}")
        if pd.notna(row['Publisher']) and row['Publisher']:
            parts.append(f"Publisher: {row['Publisher']}")
        return " | ".join(parts) if parts else ""

    df_filtered['input_text'] = df_filtered.apply(create_input_text, axis=1)

    # 入力テキストが空のレコードを除外
    df_filtered = df_filtered[df_filtered['input_text'] != ""].copy()
    print(f"入力テキストありのレコード数: {len(df_filtered):,}")

    # ラベルをID化
    label_to_id = {label: idx for idx, label in enumerate(sorted(viable_label_set))}
    id_to_label = {idx: label for label, idx in label_to_id.items()}

    print(f"\nラベル数: {len(label_to_id)}")

    # マルチホットエンコーディング
    def labels_to_multihot(labels):
        multihot = [0] * len(label_to_id)
        for label in labels:
            if label in label_to_id:
                multihot[label_to_id[label]] = 1
        return multihot

    df_filtered['label_ids'] = df_filtered['rvk_labels'].apply(
        lambda labels: [label_to_id[label] for label in labels if label in label_to_id]
    )

    # train/val/testに分割（80/10/10）
    print("\nデータを分割中...")
    train_df, temp_df = train_test_split(df_filtered, test_size=0.2, random_state=42)
    val_df, test_df = train_test_split(temp_df, test_size=0.5, random_state=42)

    print(f"Train: {len(train_df):,} レコード")
    print(f"Val: {len(val_df):,} レコード")
    print(f"Test: {len(test_df):,} レコード")

    # データを保存
    output_dir = Path("processed_data")
    output_dir.mkdir(exist_ok=True)

    print(f"\n処理済みデータを保存中: {output_dir}")

    # 必要なカラムのみ保存
    columns_to_save = ['input_text', 'rvk_labels', 'label_ids', 'num_labels']

    train_df[columns_to_save].to_json(
        output_dir / "train.jsonl",
        orient='records',
        lines=True,
        force_ascii=False
    )
    val_df[columns_to_save].to_json(
        output_dir / "val.jsonl",
        orient='records',
        lines=True,
        force_ascii=False
    )
    test_df[columns_to_save].to_json(
        output_dir / "test.jsonl",
        orient='records',
        lines=True,
        force_ascii=False
    )

    # ラベルマッピングを保存
    with open(output_dir / "label_mapping.json", 'w', encoding='utf-8') as f:
        json.dump({
            'label_to_id': label_to_id,
            'id_to_label': id_to_label,
            'num_labels': len(label_to_id)
        }, f, ensure_ascii=False, indent=2)

    # フィルタリング後のラベル統計を保存
    filtered_label_counter = Counter()
    for labels in df_filtered['rvk_labels']:
        filtered_label_counter.update(labels)

    label_stats = pd.DataFrame([
        {'label': label, 'count': count, 'label_id': label_to_id[label]}
        for label, count in filtered_label_counter.most_common()
    ])
    label_stats.to_csv(output_dir / "label_stats.csv", index=False)

    print(f"\n完了! 以下のファイルが生成されました:")
    print(f"  - {output_dir / 'train.jsonl'}")
    print(f"  - {output_dir / 'val.jsonl'}")
    print(f"  - {output_dir / 'test.jsonl'}")
    print(f"  - {output_dir / 'label_mapping.json'}")
    print(f"  - {output_dir / 'label_stats.csv'}")

    print("\n" + "=" * 80)
    print("データ統計")
    print("=" * 80)
    print(f"ラベル数: {len(label_to_id):,}")
    print(f"Train: {len(train_df):,} ({len(train_df)/len(df_filtered)*100:.1f}%)")
    print(f"Val: {len(val_df):,} ({len(val_df)/len(df_filtered)*100:.1f}%)")
    print(f"Test: {len(test_df):,} ({len(test_df)/len(df_filtered)*100:.1f}%)")
    print(f"平均ラベル数/レコード: {df_filtered['num_labels'].mean():.2f}")
    print(f"最小ラベル数: {df_filtered['num_labels'].min()}")
    print(f"最大ラベル数: {df_filtered['num_labels'].max()}")

    # 各分割でのラベル分布も確認
    print("\n各分割のラベル統計:")
    for name, split_df in [('Train', train_df), ('Val', val_df), ('Test', test_df)]:
        split_labels = []
        for labels in split_df['rvk_labels']:
            split_labels.extend(labels)
        unique_in_split = len(set(split_labels))
        print(f"  {name}: {unique_in_split:,}個のユニークラベル ({unique_in_split/len(label_to_id)*100:.1f}%)")

if __name__ == "__main__":
    main()
