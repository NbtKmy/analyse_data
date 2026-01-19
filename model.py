"""
XLM-RoBERTaベースのマルチラベル分類モデル
"""

import torch
import torch.nn as nn
from transformers import XLMRobertaModel, XLMRobertaTokenizer


class RVKMultiLabelClassifier(nn.Module):
    """
    XLM-RoBERTaベースのマルチラベル分類器
    """

    def __init__(self, num_labels, model_name="xlm-roberta-base", dropout=0.1):
        """
        Args:
            num_labels: ラベルの数
            model_name: 使用するXLM-RoBERTaモデル名
            dropout: ドロップアウト率
        """
        super().__init__()
        self.num_labels = num_labels

        # XLM-RoBERTaモデルをロード
        self.roberta = XLMRobertaModel.from_pretrained(model_name)

        # 分類ヘッド
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(self.roberta.config.hidden_size, num_labels)

    def forward(self, input_ids, attention_mask):
        """
        前向き伝播

        Args:
            input_ids: トークンID [batch_size, seq_length]
            attention_mask: アテンションマスク [batch_size, seq_length]

        Returns:
            logits: 各ラベルのロジット [batch_size, num_labels]
        """
        # XLM-RoBERTaで特徴抽出
        outputs = self.roberta(
            input_ids=input_ids,
            attention_mask=attention_mask
        )

        # [CLS]トークンの出力を使用
        pooled_output = outputs.last_hidden_state[:, 0, :]  # [batch_size, hidden_size]

        # ドロップアウトと分類
        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)  # [batch_size, num_labels]

        return logits


def load_tokenizer(model_name="xlm-roberta-base"):
    """
    トークナイザーをロード

    Args:
        model_name: 使用するXLM-RoBERTaモデル名

    Returns:
        tokenizer: XLM-RoBERTaトークナイザー
    """
    return XLMRobertaTokenizer.from_pretrained(model_name)


def get_device():
    """
    利用可能なデバイスを自動選択

    Returns:
        device: CUDA > MPS > CPUの順で利用可能なデバイス
    """
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif torch.backends.mps.is_available():
        return torch.device("mps")
    else:
        return torch.device("cpu")


def get_model(num_labels, model_name="xlm-roberta-base", dropout=0.1, device=None):
    """
    モデルを作成してデバイスに移動

    Args:
        num_labels: ラベルの数
        model_name: 使用するXLM-RoBERTaモデル名
        dropout: ドロップアウト率
        device: 使用するデバイス (Noneの場合は自動選択)

    Returns:
        model: 初期化されたモデル
    """
    if device is None:
        device = get_device()

    model = RVKMultiLabelClassifier(
        num_labels=num_labels,
        model_name=model_name,
        dropout=dropout
    )
    model = model.to(device)
    return model


if __name__ == "__main__":
    # テスト
    print("モデルのテスト...")

    device = get_device()
    print(f"使用デバイス: {device}")

    # ダミーデータでテスト
    num_labels = 100
    batch_size = 4
    seq_length = 128

    model = get_model(num_labels=num_labels, device=device)
    print(f"\nモデルのパラメータ数: {sum(p.numel() for p in model.parameters()):,}")

    # ダミー入力
    input_ids = torch.randint(0, 250000, (batch_size, seq_length)).to(device)
    attention_mask = torch.ones(batch_size, seq_length).to(device)

    # 前向き伝播
    with torch.no_grad():
        logits = model(input_ids, attention_mask)

    print(f"入力形状: input_ids={input_ids.shape}, attention_mask={attention_mask.shape}")
    print(f"出力形状: logits={logits.shape}")
    print("✓ モデルは正常に動作しています")
