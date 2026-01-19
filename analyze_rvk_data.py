"""
RVKデータの基本分析（marimoノートブック）
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
    return Counter, Path, ast, mo, pd, px


@app.cell
def _(mo):
    mo.md("""
    # RVKデータの分析

    このノートブックでは、XLM-Robertaでマルチレーベル分類を行うための
    RVKデータの基本的な分析を行います。

    **分析の目的:**
    - 入力: 著者（author）、タイトル（consolidated_title）、出版社（Publisher）
    - 出力: RVK-Notation（unique_rvk_notations）
    """)
    return


@app.cell
def _(Path, pd):
    # データの読み込み
    DATA_DIR = Path("dev_data")
    csv_files = sorted(DATA_DIR.glob("rvk_chunk_*.csv"))

    print(f"発見されたCSVファイル数: {len(csv_files)}")

    # 全データを結合
    dfs = []
    for csv_file in csv_files:
        df_chunk = pd.read_csv(csv_file)
        dfs.append(df_chunk)

    df = pd.concat(dfs, ignore_index=True)
    print(f"総レコード数: {len(df):,}")
    return (df,)


@app.cell
def _(df, mo):
    mo.md(f"""
    ## 基本統計

    - **総レコード数**: {len(df):,}
    - **カラム数**: {len(df.columns)}
    """)
    return


@app.cell
def _(df):
    # データの最初の数行を表示
    df.head(10)
    return


@app.cell
def _(df, mo, pd):
    # 欠損値の確認
    missing_stats = pd.DataFrame({
        'カラム': df.columns,
        '欠損数': df.isnull().sum().values,
        '欠損率(%)': (df.isnull().sum().values / len(df) * 100).round(2)
    })

    mo.md(f"""
    ## 欠損値の確認

    {mo.as_html(missing_stats)}
    """)
    return


@app.cell
def _(ast, df, mo, pd):
    # RVK表記の解析
    def parse_rvk_notations(notation_str):
        """RVK表記の文字列をパース"""
        try:
            if pd.isna(notation_str) or notation_str == '[]':
                return []
            notations = ast.literal_eval(notation_str)
            # タプルのリストから、最初の要素（RVKコード）のみを抽出
            return [notation[0] if isinstance(notation, tuple) else notation for notation in notations]
        except:
            return []

    df['rvk_labels'] = df['unique_rvk_notations'].apply(parse_rvk_notations)
    df['num_labels'] = df['rvk_labels'].apply(len)

    mo.md(f"""
    ## RVK表記の分析

    - **ラベルが付いているレコード数**: {(df['num_labels'] > 0).sum():,}
    - **ラベルが付いていないレコード数**: {(df['num_labels'] == 0).sum():,}
    - **ラベルなしの割合**: {(df['num_labels'] == 0).sum() / len(df) * 100:.2f}%
    """)
    return


@app.cell
def _(df, mo):
    # ラベル数の統計
    label_stats = df['num_labels'].describe()
    mo.md(f"""
    ### 1レコードあたりのRVKラベル数の統計

    - **平均**: {label_stats['mean']:.2f}
    - **中央値**: {label_stats['50%']:.2f}
    - **最小値**: {int(label_stats['min'])}
    - **最大値**: {int(label_stats['max'])}
    - **標準偏差**: {label_stats['std']:.2f}
    """)
    return


@app.cell
def _(df, px):
    # ラベル数の分布をプロット
    label_count_dist = df['num_labels'].value_counts().sort_index().head(20)
    fig_label_dist = px.bar(
        x=label_count_dist.index,
        y=label_count_dist.values,
        labels={'x': 'ラベル数', 'y': 'レコード数'},
        title='ラベル数の分布（上位20）'
    )
    fig_label_dist.update_layout(xaxis_type='category')
    fig_label_dist
    return


@app.cell
def _(Counter, df, mo):
    # ユニークなRVKラベルの分析
    all_labels = []
    for labels in df['rvk_labels']:
        all_labels.extend(labels)

    unique_labels = set(all_labels)
    label_counter = Counter(all_labels)

    mo.md(f"""
    ### ユニークなRVKラベル

    - **ユニークなRVKラベル数**: {len(unique_labels):,}
    - **総ラベル出現数**: {len(all_labels):,}
    """)
    return (label_counter,)


@app.cell
def _(label_counter, pd, px):
    # 最も頻出するRVKラベルをプロット
    top_labels = label_counter.most_common(30)
    top_labels_df = pd.DataFrame(top_labels, columns=['RVKラベル', '出現回数'])

    fig_top_labels = px.bar(
        top_labels_df,
        x='出現回数',
        y='RVKラベル',
        orientation='h',
        title='最も頻出するRVKラベル（上位30）'
    )
    fig_top_labels.update_layout(height=800, yaxis={'categoryorder': 'total ascending'})
    fig_top_labels
    return


@app.cell
def _(mo):
    # 入力特徴量の分析
    mo.md("""
    ## 入力特徴量の分析

    モデルへの入力として使用する特徴量の分析
    """)
    return


@app.cell
def _(df, mo):
    # タイトルの分析
    df['title_length'] = df['consolidated_title'].fillna('').apply(len)
    title_stats = df['title_length'].describe()

    mo.md(f"""
    ### タイトル（consolidated_title）

    - **欠損値**: {df['consolidated_title'].isnull().sum():,} ({df['consolidated_title'].isnull().sum() / len(df) * 100:.2f}%)
    - **ユニーク数**: {df['consolidated_title'].nunique():,}
    - **平均文字数**: {title_stats['mean']:.2f}
    - **中央値**: {title_stats['50%']:.0f}
    - **最小値**: {int(title_stats['min'])}
    - **最大値**: {int(title_stats['max'])}
    """)
    return


@app.cell
def _(df, px):
    # タイトル長の分布
    fig_title_length = px.histogram(
        df,
        x='title_length',
        nbins=50,
        title='タイトルの文字数分布',
        labels={'title_length': 'タイトル文字数', 'count': 'レコード数'}
    )
    fig_title_length.update_layout(bargap=0.1)
    fig_title_length
    return


@app.cell
def _(df, mo):
    # 著者の分析
    df['author_length'] = df['author'].fillna('').apply(len)
    author_stats = df['author_length'].describe()

    mo.md(f"""
    ### 著者（author）

    - **欠損値**: {df['author'].isnull().sum():,} ({df['author'].isnull().sum() / len(df) * 100:.2f}%)
    - **ユニーク数**: {df['author'].nunique():,}
    - **平均文字数**: {author_stats['mean']:.2f}
    - **中央値**: {author_stats['50%']:.0f}
    """)
    return


@app.cell
def _(df, mo):
    # 出版社の分析
    mo.md(f"""
    ### 出版社（Publisher）

    - **欠損値**: {df['Publisher'].isnull().sum():,} ({df['Publisher'].isnull().sum() / len(df) * 100:.2f}%)
    - **ユニーク数**: {df['Publisher'].nunique():,}
    """)
    return


@app.cell
def _(df, pd, px):
    # 最も多い出版社
    top_publishers = df['Publisher'].value_counts().head(20)
    top_publishers_df = pd.DataFrame({
        '出版社': top_publishers.index,
        'レコード数': top_publishers.values
    })

    fig_publishers = px.bar(
        top_publishers_df,
        x='レコード数',
        y='出版社',
        orientation='h',
        title='最も多い出版社（上位20）'
    )
    fig_publishers.update_layout(height=600, yaxis={'categoryorder': 'total ascending'})
    fig_publishers
    return


@app.cell
def _(df, mo):
    # データ品質の分析
    has_title = ~df['consolidated_title'].isnull()
    has_author = ~df['author'].isnull()
    has_publisher = ~df['Publisher'].isnull()
    has_labels = df['num_labels'] > 0

    complete_data = has_title & has_author & has_publisher & has_labels
    usable_data = has_labels & (has_title | has_author | has_publisher)

    mo.md(f"""
    ## データ品質の分析

    ### 完全性

    - **完全なデータ（全て揃っている）**: {complete_data.sum():,} ({complete_data.sum() / len(df) * 100:.2f}%)
    - **使用可能なデータ（ラベルがあり、少なくとも1つの入力特徴量がある）**: {usable_data.sum():,} ({usable_data.sum() / len(df) * 100:.2f}%)
    """)
    return complete_data, has_author, has_labels, has_publisher, has_title


@app.cell
def _(complete_data, has_author, has_labels, has_publisher, has_title, pd, px):
    # 入力特徴量の組み合わせ
    combinations = pd.DataFrame({
        '組み合わせ': [
            'タイトルのみ',
            '著者のみ',
            '出版社のみ',
            'タイトル+著者',
            'タイトル+出版社',
            '著者+出版社',
            '全て揃っている'
        ],
        'レコード数': [
            (has_title & ~has_author & ~has_publisher & has_labels).sum(),
            (~has_title & has_author & ~has_publisher & has_labels).sum(),
            (~has_title & ~has_author & has_publisher & has_labels).sum(),
            (has_title & has_author & ~has_publisher & has_labels).sum(),
            (has_title & ~has_author & has_publisher & has_labels).sum(),
            (~has_title & has_author & has_publisher & has_labels).sum(),
            complete_data.sum()
        ]
    })

    fig_combinations = px.bar(
        combinations,
        x='レコード数',
        y='組み合わせ',
        orientation='h',
        title='入力特徴量の組み合わせ（ラベルありのみ）'
    )
    fig_combinations.update_layout(yaxis={'categoryorder': 'total ascending'})
    fig_combinations
    return


@app.cell
def _(complete_data, df, label_counter, mo):
    # 分析結果の要約
    mo.md(f"""
    ## 分析結果の要約

    ### 全体統計
    - 総レコード数: **{len(df):,}**
    - ラベル付きレコード数: **{(df['num_labels'] > 0).sum():,}**
    - ユニークなRVKラベル数: **{len(label_counter):,}**
    - 1レコードあたりの平均ラベル数: **{df['num_labels'].mean():.2f}**
    - 完全なレコード数（全入力特徴量+ラベルあり）: **{complete_data.sum():,}**

    ### モデル構築への示唆
    1. **マルチレーベル分類**: 平均的に各レコードには複数のRVKラベルが付いている
    2. **クラス数**: {len(label_counter):,}個のユニークなRVKラベルがあり、大規模な多クラス分類問題
    3. **データの完全性**: 完全なデータは{complete_data.sum() / len(df) * 100:.2f}%で、欠損値の処理が必要
    4. **テキスト入力**: タイトル、著者、出版社を結合してXLM-Robertaへの入力とする
    """)
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
