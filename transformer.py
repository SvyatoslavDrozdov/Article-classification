import torch
import torch.nn as nn
import torch.nn.functional as F


class MultiheadSelfAttention(nn.Module):
    def __init__(self, d_model: int, num_heads: int, dropout: float) -> None:
        super().__init__()
        self.d_model: int = d_model
        self.num_heads: int = num_heads
        self.head_dim: int = d_model // num_heads

        self.Q_transform: nn.Linear = nn.Linear(d_model, d_model)
        self.K_transform: nn.Linear = nn.Linear(d_model, d_model)
        self.V_transform: nn.Linear = nn.Linear(d_model, d_model)

        self.linear_layer: nn.Linear = nn.Linear(d_model, d_model)
        self.dropout: nn.Dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, attention_mask: torch.Tensor | None = None) -> torch.Tensor:
        batch_size, token_count, d_model = x.shape
        query = self.Q_transform(x)
        key = self.K_transform(x)
        value = self.V_transform(x)

        query = query.view(batch_size, token_count, self.num_heads, self.head_dim).transpose(1, 2)
        key = key.view(batch_size, token_count, self.num_heads, self.head_dim).transpose(1, 2)
        value = value.view(batch_size, token_count, self.num_heads, self.head_dim).transpose(1, 2)

        scores: torch.Tensor = torch.matmul(query, key.transpose(2, 3)) / self.head_dim ** 0.5
        if attention_mask is not None:
            mask = attention_mask[:, None, None,:]
            scores = scores.masked_fill(mask == 0, float("-inf"))
        attention_weights = F.softmax(scores, dim=-1)
        attention_weights = self.dropout(attention_weights)

        output: torch.Tensor = torch.matmul(attention_weights, value)
        output = output.transpose(1, 2).contiguous().view(batch_size, token_count, self.d_model)
        output = self.linear_layer(output)

        return output


class FeedForward(nn.Module):
    def __init__(self, d_model: int, d_feed_forward: int, dropout: float) -> None:
        super().__init__()
        self.in_linear_layer: nn.Linear = nn.Linear(d_model, d_feed_forward)
        self.GELU: nn.GELU = nn.GELU()
        self.dropout: nn.Dropout = nn.Dropout(dropout)
        self.out_linear_layer: nn.Linear = nn.Linear(d_feed_forward, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        output: torch.Tensor = self.in_linear_layer(x)
        output = self.GELU(output)
        output = self.dropout(output)
        output = self.out_linear_layer(output)

        return output


class EncoderBlock(nn.Module):
    def __init__(self, d_model: int, num_heads: int, d_feed_forward: int, dropout: float) -> None:
        super().__init__()
        self.layer_norm_1: nn.LayerNorm = nn.LayerNorm(d_model)
        self.attention = MultiheadSelfAttention(d_model, num_heads, dropout)
        self.dropout_1 = nn.Dropout(dropout)

        self.layer_norm_2 = nn.LayerNorm(d_model)
        self.feed_forward = FeedForward(d_model, d_feed_forward, dropout)
        self.dropout_2 = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, attention_mask=None) -> torch.Tensor:
        output_1 = self.layer_norm_1(x)
        output_1 = self.attention(output_1, attention_mask)
        output_1 = self.dropout_1(output_1)
        output_1 = x + output_1

        output_2 = self.layer_norm_2(output_1)
        output_2 = self.feed_forward(output_2)
        output_2 = self.dropout_2(output_2)

        output = output_1 + output_2
        return output


class Transformer(nn.Module):
    def __init__(self, vocab_size: int, max_len: int, num_classes: int, d_model: int = 128, num_heads: int = 4,
                 num_layers: int = 2, d_feed_forward: int = 512, dropout: float = 0.1, pad_token_id: int = 0) -> None:
        super().__init__()
        self.pad_token_id: int = pad_token_id
        self.max_len: int = max_len
        self.d_model: int = d_model

        self.token_embedding: nn.Embedding = nn.Embedding(vocab_size, d_model, padding_idx=pad_token_id)
        self.position_embedding: nn.Embedding = nn.Embedding(max_len, d_model)
        self.dropout: nn.Dropout = nn.Dropout(dropout)

        self.layers: nn.ModuleList = nn.ModuleList([
            EncoderBlock(d_model, num_heads, d_feed_forward, dropout)
            for _ in range(num_layers)
        ])

        self.layer_norm: nn.LayerNorm = nn.LayerNorm(d_model)
        self.linear_layer: nn.Linear = nn.Linear(d_model, num_classes)

    @staticmethod
    def masked_mean_pooling(x: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        mask: torch.Tensor = attention_mask.unsqueeze(-1)
        x = x * mask
        x_sum: torch.Tensor = x.sum(dim=1)
        tokens_count: torch.Tensor = mask.sum(dim=1)
        return x_sum / tokens_count

    def forward(self, inputs_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        batch_size, token_count = inputs_ids.shape
        positions: torch.Tensor = torch.arange(token_count, device=inputs_ids.device).unsqueeze(0)
        x = self.token_embedding(inputs_ids) + self.position_embedding(positions)
        x = self.dropout(x)
        for layer in self.layers:
            x = layer(x, attention_mask)
        x = self.layer_norm(x)
        pooled: torch.Tensor = self.masked_mean_pooling(x, attention_mask)
        logits: torch.Tensor = self.linear_layer(pooled)
        return logits
