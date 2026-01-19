"""
RVKラベル分布の詳細分析
実務での実現可能性を評価するための追加分析
"""

import marimo

__generated_with = "0.19.4"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    import pandas as pd
    import numpy as np
    from pathlib import Path
    import ast
    from collections import Counter
    import plotly.express as px
    import plotly.graph_objects as go
    return Counter, Path, ast, go, mo, np, pd, px


@app.cell
def _(mo):
    mo.md("""
    # RVKラベル分布の詳細分析

    XLM-Robertaで実務レベルの結果を出せるかを評価するための分析
    """)
    return


@app.cell
def _(Path, ast, pd):
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

    print(f"ラベル付きレコード数: {len(df_with_labels):,}")
    return df, df_with_labels, parse_rvk_notations


@app.cell
def _(Counter, df_with_labels, mo):
    # 全ラベルを抽出
    all_labels = []
    for labels in df_with_labels['rvk_labels']:
        all_labels.extend(labels)

    label_counter = Counter(all_labels)

    mo.md(f"""
    ## 基本統計（再確認）

    - **総ラベル出現数**: {len(all_labels):,}
    - **ユニークラベル数**: {len(label_counter):,}
    - **平均出現回数**: {len(all_labels) / len(label_counter):.2f}
    """)
    return all_labels, label_counter


@app.cell
def _(label_counter, mo, np, pd, px):
    # ラベルごとのサンプル数分布
    label_counts = pd.DataFrame({
        'label': list(label_counter.keys()),
        'count': list(label_counter.values())
    }).sort_values('count', ascending=False).reset_index(drop=True)

    # 統計
    count_stats = label_counts['count'].describe()

    mo.md(f"""
    ## ラベルごとのサンプル数統計

    - **最大**: {int(count_stats['max'])}回
    - **75%点**: {int(count_stats['75%'])}回
    - **中央値**: {int(count_stats['50%'])}回
    - **25%点**: {int(count_stats['25%'])}回
    - **最小**: {int(count_stats['min'])}回
    - **平均**: {count_stats['mean']:.2f}回
    - **標準偏差**: {count_stats['std']:.2f}

    ### 🚨 **重大な問題点**
    中央値が{int(count_stats['50%'])}回ということは、**半数のラベルが{int(count_stats['50%'])}回以下しか出現していない**
    """)
    return count_stats, label_counts


@app.cell
def _(label_counts, px):
    # 上位100ラベルの分布
    fig_top100 = px.bar(
        label_counts.head(100),
        x=label_counts.head(100).index,
        y='count',
        title='上位100ラベルの出現回数',
        labels={'index': 'ラベルランク', 'count': '出現回数'}
    )
    fig_top100.update_layout(showlegend=False)
    fig_top100
    return (fig_top100,)


@app.cell
def _(label_counts, mo, np, px):
    # Long-tail分布の可視化（対数スケール）
    fig_longtail = px.line(
        label_counts,
        x=label_counts.index,
        y='count',
        title='ラベル分布（Long-tail分析）',
        labels={'index': 'ラベルランク', 'count': '出現回数'},
        log_y=True
    )
    fig_longtail.add_hline(y=10, line_dash="dash", line_color="red",
                           annotation_text="10サンプル")
    fig_longtail.add_hline(y=50, line_dash="dash", line_color="orange",
                           annotation_text="50サンプル")
    fig_longtail.add_hline(y=100, line_dash="dash", line_color="green",
                           annotation_text="100サンプル")
    fig_longtail

    # サンプル数ごとのラベル数
    threshold_analysis = {
        'サンプル数が1回のみ': (label_counts['count'] == 1).sum(),
        'サンプル数が2-4回': ((label_counts['count'] >= 2) & (label_counts['count'] <= 4)).sum(),
        'サンプル数が5-9回': ((label_counts['count'] >= 5) & (label_counts['count'] <= 9)).sum(),
        'サンプル数が10-49回': ((label_counts['count'] >= 10) & (label_counts['count'] < 50)).sum(),
        'サンプル数が50-99回': ((label_counts['count'] >= 50) & (label_counts['count'] < 100)).sum(),
        'サンプル数が100回以上': (label_counts['count'] >= 100).sum(),
    }

    mo.md(f"""
    ## Long-tail分析

    {', '.join([f"**{k}**: {v:,}個 ({v/len(label_counts)*100:.1f}%)" for k, v in threshold_analysis.items()])}
    """)
    return fig_longtail, threshold_analysis


@app.cell
def _(label_counts, mo):
    # 実用的なラベル数の評価
    min_samples_thresholds = [5, 10, 20, 50, 100]

    viable_labels = {}
    for threshold in min_samples_thresholds:
        viable_labels[threshold] = (label_counts['count'] >= threshold).sum()

    mo.md(f"""
    ## 実用的なラベル数の評価

    機械学習では、各ラベルに最低限のサンプル数が必要です：

    {chr(10).join([f"- **最低{threshold}サンプル以上**: {count:,}ラベル ({count/len(label_counts)*100:.1f}%)"
                   for threshold, count in viable_labels.items()])}
    """)
    return min_samples_thresholds, viable_labels


@app.cell
def _(label_counts):
    # RVK階層構造の分析
    # RVK表記は階層的（例: "AP 50300" → "AP" が主分類）

    def extract_main_category(rvk_code):
        """RVKコードから主分類を抽出"""
        if not isinstance(rvk_code, str):
            return None
        # スペースまたは数字の前までを主分類とする
        parts = rvk_code.strip().split()
        if parts:
            # 最初の文字列から数字を除去
            main = ''.join([c for c in parts[0] if c.isalpha()])
            return main if main else None
        return None

    label_counts['main_category'] = label_counts['label'].apply(extract_main_category)

    main_category_stats = label_counts.groupby('main_category').agg({
        'count': ['sum', 'count', 'mean']
    }).round(2)

    main_category_stats.columns = ['総出現数', 'サブラベル数', '平均出現数']
    main_category_stats = main_category_stats.sort_values('総出現数', ascending=False)

    main_category_stats.head(20)
    return extract_main_category, main_category_stats


@app.cell
def _(mo):
    mo.md("""
    ## RVK階層構造の活用

    上の表は、RVKコードの主分類（先頭のアルファベット部分）ごとの統計です。
    階層的分類アプローチを使うことで、問題を簡略化できる可能性があります。
    """)
    return


@app.cell
def _(label_counts, mo):
    # 推奨事項
    mo.md(f"""
    ## 実務での実現可能性の評価

    ### 🔴 **現状のまま全27,051ラベルで学習**: 困難

    **問題点:**
    1. データ不足のラベルが多すぎる（半数が{int(label_counts['count'].median())}回以下）
    2. モデルが複雑すぎてメモリ不足の可能性
    3. 学習時間が非常に長い
    4. 低頻度ラベルの性能が低い

    ### 🟡 **推奨アプローチ1: ラベルのフィルタリング**

    最低サンプル数（例: 50回以上）でフィルタリング
    - **50回以上**: {(label_counts['count'] >= 50).sum():,}ラベル
    - **100回以上**: {(label_counts['count'] >= 100).sum():,}ラベル

    → より実用的なモデルサイズとデータ量のバランス

    ### 🟢 **推奨アプローチ2: 階層的分類**

    1. **第1段階**: 主分類（粗いカテゴリ）を予測
    2. **第2段階**: 主分類内での詳細分類を予測

    → 問題を分割して解決しやすくする

    ### 🟢 **推奨アプローチ3: Few-shot学習**

    低頻度ラベルには、Few-shot学習やメタ学習の技術を使用

    ### 💡 **次のステップ**

    1. **最低サンプル数の閾値を決定**（50-100を推奨）
    2. **フィルタリング後のデータで実験**
    3. **階層構造の活用を検討**
    4. **評価指標の定義**（Precision@K, Recall@K, F1-scoreなど）
    """)
    return


@app.cell
def _(df_with_labels, label_counts, mo, pd):
    # フィルタリングシミュレーション
    min_samples = 50
    viable_label_set = set(label_counts[label_counts['count'] >= min_samples]['label'])

    # フィルタリング後のデータ数
    def has_viable_label(labels):
        return any(label in viable_label_set for label in labels)

    df_filtered = df_with_labels[df_with_labels['rvk_labels'].apply(has_viable_label)].copy()

    # フィルタリング後のラベル（viable_label_setに含まれるもののみ）
    def filter_labels(labels):
        return [label for label in labels if label in viable_label_set]

    df_filtered['filtered_labels'] = df_filtered['rvk_labels'].apply(filter_labels)
    df_filtered['num_filtered_labels'] = df_filtered['filtered_labels'].apply(len)

    mo.md(f"""
    ## フィルタリングシミュレーション（最低{min_samples}サンプル）

    - **残るラベル数**: {len(viable_label_set):,} / 27,051 ({len(viable_label_set)/27051*100:.1f}%)
    - **残るレコード数**: {len(df_filtered):,} / {len(df_with_labels):,} ({len(df_filtered)/len(df_with_labels)*100:.1f}%)
    - **平均ラベル数/レコード**: {df_filtered['num_filtered_labels'].mean():.2f}

    → これでも実用的なデータセットとして使用可能
    """)
    return df_filtered, filter_labels, has_viable_label, min_samples, viable_label_set


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
