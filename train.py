"""
RVKマルチラベル分類のトレーニングスクリプト
"""

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from transformers import get_linear_schedule_with_warmup
import json
from pathlib import Path
import numpy as np
from tqdm import tqdm
from model import get_model, load_tokenizer, get_device
from sklearn.metrics import f1_score, precision_score, recall_score
import argparse


class RVKDataset(Dataset):
    """RVKマルチラベル分類用のデータセット"""

    def __init__(self, data_path, tokenizer, max_length=256):
        """
        Args:
            data_path: JSONLファイルのパス
            tokenizer: トークナイザー
            max_length: 最大トークン長
        """
        self.data = []
        with open(data_path, 'r', encoding='utf-8') as f:
            for line in f:
                self.data.append(json.loads(line))

        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]

        # トークナイズ
        encoding = self.tokenizer(
            item['input_text'],
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )

        # マルチホットエンコーディング（label_idsをテンソルに変換）
        label_ids = item['label_ids']

        return {
            'input_ids': encoding['input_ids'].squeeze(0),
            'attention_mask': encoding['attention_mask'].squeeze(0),
            'label_ids': label_ids
        }


def collate_fn(batch):
    """
    バッチをまとめる関数
    """
    input_ids = torch.stack([item['input_ids'] for item in batch])
    attention_mask = torch.stack([item['attention_mask'] for item in batch])

    # ラベルIDのリストをマルチホットベクトルに変換
    # まず、最大のラベルIDを見つける
    max_label_id = max(max(item['label_ids']) for item in batch if item['label_ids'])

    # マルチホットベクトルを作成
    labels = torch.zeros(len(batch), max_label_id + 1)
    for i, item in enumerate(batch):
        for label_id in item['label_ids']:
            labels[i, label_id] = 1.0

    return {
        'input_ids': input_ids,
        'attention_mask': attention_mask,
        'labels': labels
    }


def compute_metrics(logits, labels, k_values=[1, 3, 5, 10]):
    """
    評価メトリクスを計算

    Args:
        logits: モデルの出力 [batch_size, num_labels]
        labels: 正解ラベル [batch_size, num_labels]
        k_values: Recall@Kを計算するKの値

    Returns:
        metrics: 評価メトリクスの辞書
    """
    # sigmoidを適用して確率に変換
    probs = torch.sigmoid(logits)

    # 予測（0.5を閾値とする）
    preds = (probs > 0.5).float()

    # numpyに変換
    preds_np = preds.cpu().numpy()
    labels_np = labels.cpu().numpy()
    probs_np = probs.cpu().numpy()

    metrics = {}

    # Precision, Recall, F1 (マイクロ平均)
    if preds_np.sum() > 0:  # 予測がある場合のみ計算
        metrics['precision_micro'] = precision_score(labels_np, preds_np, average='micro', zero_division=0)
        metrics['recall_micro'] = recall_score(labels_np, preds_np, average='micro', zero_division=0)
        metrics['f1_micro'] = f1_score(labels_np, preds_np, average='micro', zero_division=0)
    else:
        metrics['precision_micro'] = 0.0
        metrics['recall_micro'] = 0.0
        metrics['f1_micro'] = 0.0

    # Recall@K
    for k in k_values:
        recall_at_k = compute_recall_at_k(probs_np, labels_np, k)
        metrics[f'recall@{k}'] = recall_at_k

    return metrics


def compute_recall_at_k(probs, labels, k):
    """
    Recall@Kを計算

    Args:
        probs: 予測確率 [batch_size, num_labels]
        labels: 正解ラベル [batch_size, num_labels]
        k: 上位K個

    Returns:
        recall_at_k: Recall@K
    """
    # 各サンプルについてトップK個の予測を取得
    top_k_indices = np.argsort(probs, axis=1)[:, -k:]

    # Recall@Kを計算
    total_relevant = 0
    total_retrieved_relevant = 0

    for i in range(len(labels)):
        # 正解ラベルのインデックス
        true_indices = np.where(labels[i] == 1)[0]

        if len(true_indices) == 0:
            continue

        # トップKの中に正解ラベルがいくつ含まれているか
        retrieved_relevant = len(set(top_k_indices[i]) & set(true_indices))

        total_relevant += len(true_indices)
        total_retrieved_relevant += retrieved_relevant

    if total_relevant == 0:
        return 0.0

    return total_retrieved_relevant / total_relevant


def train_epoch(model, dataloader, optimizer, scheduler, device, num_labels):
    """
    1エポックのトレーニング
    """
    model.train()
    total_loss = 0
    all_logits = []
    all_labels = []

    criterion = nn.BCEWithLogitsLoss()

    progress_bar = tqdm(
        dataloader,
        desc="  Training",
        position=1,
        leave=True,
        ncols=100
    )

    for batch in progress_bar:
        # デバイスに移動
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['labels'].to(device)

        # ラベルのサイズをnum_labelsに合わせる
        if labels.shape[1] < num_labels:
            padding = torch.zeros(labels.shape[0], num_labels - labels.shape[1]).to(device)
            labels = torch.cat([labels, padding], dim=1)

        # 前向き伝播
        optimizer.zero_grad()
        logits = model(input_ids, attention_mask)

        # 損失計算
        loss = criterion(logits, labels)

        # 逆伝播
        loss.backward()
        optimizer.step()
        scheduler.step()

        total_loss += loss.item()

        # メトリクス計算用に保存
        all_logits.append(logits.detach())
        all_labels.append(labels.detach())

        progress_bar.set_postfix({'loss': f'{loss.item():.4f}'})

    # 全バッチのメトリクスを計算
    all_logits = torch.cat(all_logits, dim=0)
    all_labels = torch.cat(all_labels, dim=0)
    metrics = compute_metrics(all_logits, all_labels)

    metrics['loss'] = total_loss / len(dataloader)

    return metrics


def evaluate(model, dataloader, device, num_labels):
    """
    評価
    """
    model.eval()
    total_loss = 0
    all_logits = []
    all_labels = []

    criterion = nn.BCEWithLogitsLoss()

    progress_bar = tqdm(
        dataloader,
        desc="  Evaluating",
        position=1,
        leave=True,
        ncols=100
    )

    with torch.no_grad():
        for batch in progress_bar:
            # デバイスに移動
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)

            # ラベルのサイズをnum_labelsに合わせる
            if labels.shape[1] < num_labels:
                padding = torch.zeros(labels.shape[0], num_labels - labels.shape[1]).to(device)
                labels = torch.cat([labels, padding], dim=1)

            # 前向き伝播
            logits = model(input_ids, attention_mask)

            # 損失計算
            loss = criterion(logits, labels)
            total_loss += loss.item()

            # メトリクス計算用に保存
            all_logits.append(logits)
            all_labels.append(labels)

    # 全バッチのメトリクスを計算
    all_logits = torch.cat(all_logits, dim=0)
    all_labels = torch.cat(all_labels, dim=0)
    metrics = compute_metrics(all_logits, all_labels)

    metrics['loss'] = total_loss / len(dataloader)

    return metrics


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, default='processed_data')
    parser.add_argument('--output_dir', type=str, default='outputs')
    parser.add_argument('--model_name', type=str, default='xlm-roberta-base')
    parser.add_argument('--max_length', type=int, default=256)
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--epochs', type=int, default=5)
    parser.add_argument('--lr', type=float, default=2e-5)
    parser.add_argument('--warmup_ratio', type=float, default=0.1)
    parser.add_argument('--dropout', type=float, default=0.1)
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()

    # シード設定
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    # デバイス (CUDA > MPS > CPUの順で自動選択)
    device = get_device()
    print(f"使用デバイス: {device}")

    # データディレクトリ
    data_dir = Path(args.data_dir)

    # ラベルマッピングをロード
    with open(data_dir / 'label_mapping.json', 'r') as f:
        label_mapping = json.load(f)
    num_labels = label_mapping['num_labels']
    print(f"ラベル数: {num_labels}")

    # トークナイザーをロード
    print("トークナイザーをロード中...")
    tokenizer = load_tokenizer(args.model_name)

    # データセットを作成
    print("データセットを読み込み中...")
    train_dataset = RVKDataset(data_dir / 'train.jsonl', tokenizer, args.max_length)
    val_dataset = RVKDataset(data_dir / 'val.jsonl', tokenizer, args.max_length)
    test_dataset = RVKDataset(data_dir / 'test.jsonl', tokenizer, args.max_length)

    print(f"Train: {len(train_dataset)} サンプル")
    print(f"Val: {len(val_dataset)} サンプル")
    print(f"Test: {len(test_dataset)} サンプル")

    # DataLoaderを作成
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=collate_fn
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=collate_fn
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=collate_fn
    )

    # モデルを作成
    print("モデルを作成中...")
    model = get_model(num_labels, args.model_name, args.dropout, device)
    print(f"モデルパラメータ数: {sum(p.numel() for p in model.parameters()):,}")

    # オプティマイザとスケジューラ
    optimizer = AdamW(model.parameters(), lr=args.lr)
    total_steps = len(train_loader) * args.epochs
    warmup_steps = int(total_steps * args.warmup_ratio)
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_steps
    )

    # 出力ディレクトリ
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True)

    # トレーニング
    print("\n" + "=" * 80)
    print("トレーニング開始")
    print("=" * 80)

    best_recall_at_5 = 0.0

    # エポック全体の進捗バー
    epoch_progress = tqdm(range(args.epochs), desc="全体の進捗", position=0)

    for epoch in epoch_progress:
        print(f"\n{'='*80}")
        print(f"Epoch {epoch + 1}/{args.epochs}")
        print(f"{'='*80}")

        # トレーニング
        train_metrics = train_epoch(model, train_loader, optimizer, scheduler, device, num_labels)
        print(f"\n[Train] Loss: {train_metrics['loss']:.4f} | "
              f"F1: {train_metrics['f1_micro']:.4f} | "
              f"Recall@5: {train_metrics['recall@5']:.4f}")

        # 検証
        val_metrics = evaluate(model, val_loader, device, num_labels)
        print(f"[Val]   Loss: {val_metrics['loss']:.4f} | "
              f"F1: {val_metrics['f1_micro']:.4f} | "
              f"Recall@5: {val_metrics['recall@5']:.4f}")

        # エポック進捗バーを更新
        epoch_progress.set_postfix({
            'train_loss': f'{train_metrics["loss"]:.4f}',
            'val_loss': f'{val_metrics["loss"]:.4f}',
            'val_recall@5': f'{val_metrics["recall@5"]:.4f}'
        })

        # ベストモデルを保存
        if val_metrics['recall@5'] > best_recall_at_5:
            best_recall_at_5 = val_metrics['recall@5']
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_metrics': val_metrics,
            }, output_dir / 'best_model.pt')
            print(f"✓ ベストモデルを保存（Recall@5: {best_recall_at_5:.4f}）")

    epoch_progress.close()

    # 最後のモデルを保存
    torch.save({
        'epoch': args.epochs - 1,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
    }, output_dir / 'last_model.pt')

    # テストセットで評価
    print("\n" + "=" * 80)
    print("テストセットで評価")
    print("=" * 80)

    # ベストモデルをロード
    checkpoint = torch.load(output_dir / 'best_model.pt')
    model.load_state_dict(checkpoint['model_state_dict'])

    test_metrics = evaluate(model, test_loader, device, num_labels)
    print("\nTest Results:")
    print(f"  Loss: {test_metrics['loss']:.4f}")
    print(f"  Precision (micro): {test_metrics['precision_micro']:.4f}")
    print(f"  Recall (micro): {test_metrics['recall_micro']:.4f}")
    print(f"  F1 (micro): {test_metrics['f1_micro']:.4f}")
    print(f"  Recall@1: {test_metrics['recall@1']:.4f}")
    print(f"  Recall@3: {test_metrics['recall@3']:.4f}")
    print(f"  Recall@5: {test_metrics['recall@5']:.4f}")
    print(f"  Recall@10: {test_metrics['recall@10']:.4f}")

    # メトリクスを保存
    with open(output_dir / 'test_metrics.json', 'w') as f:
        json.dump(test_metrics, f, indent=2)

    print(f"\n完了! モデルとメトリクスは {output_dir} に保存されました")


if __name__ == "__main__":
    main()
