import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional, Tuple


class BERTEmbeddings(nn.Module):
    """
    BERT Embeddings:
        Token Embeddings: Token information
        Position Embeddings: Position information
        Segment Embeddings: Used for distinguishing between multiple input sentences
    """
    def __init__(self, vocab_size: int, embed_size: int, max_len: int, dropout_rate: float = 0.1):
        """
        :param vocab_size: total_vocab_size
        :param embed_size: embedding size of token embedding
        :param max_len: max_len of sequence
        :param dropout_rate: dropout rate
        """
        super(BERTEmbeddings, self).__init__()
        # [0] Token is used for padding, so no embedding calculation is performed.
        self.token_embeddings = nn.Embedding(vocab_size, embed_size, padding_idx=0)
        self.position_embeddings = nn.Embedding(max_len, embed_size)
        # Stores the information about which segment each token belongs to
        self.segment_embeddings = nn.Embedding(3, embed_size, padding_idx=0)
        # layer_norm + dropout
        self.layer_norm = nn.LayerNorm(embed_size, eps=1e-6)
        self.dropout = nn.Dropout(p=dropout_rate)

    def forward(self, seq: torch.Tensor, segment_label: Optional[torch.Tensor] = None) -> torch.Tensor:
        batch_size, seq_length = seq.size(0), seq.size(1)  # seq: (batch, seq_len)
        position_ids = torch.arange(seq_length, dtype=torch.long, device=seq.device)
        position_ids = position_ids.unsqueeze(0).expand(batch_size, seq_length)  # position_ids: (batch_size, seq_length)
        # token, position embeddings
        token_embeddings = self.token_embeddings(seq)
        position_embeddings = self.position_embeddings(position_ids)
        # bert_embeddings
        embeddings = token_embeddings + position_embeddings
        # segment embeddings
        if segment_label is not None:
            segment_embeddings = self.segment_embeddings(segment_label)
            embeddings += segment_embeddings
        # layer-norm + dropout
        embeddings = self.dropout(self.layer_norm(embeddings))

        return embeddings


class MultiHeadedAttention(nn.Module):
    """
    Multi-Head Attention
    """
    def __init__(self, head_num, hidden_dim, dropout_rate_attn=0.1):
        """
        :param head_num: number of attention heads
        :param hidden_dim: hidden dimension
        :param dropout_rate: dropout rate
        """
        super(MultiHeadedAttention, self).__init__()

        assert hidden_dim % head_num == 0, "Wrong hidden_dim, head_num"

        self.hidden_dim = hidden_dim
        # When V and K are the same.
        self.head_dim = hidden_dim // head_num
        self.head_num = head_num
        # Q, K, V linear layers
        self.query_linear = nn.Linear(hidden_dim, hidden_dim)
        self.key_linear = nn.Linear(hidden_dim, hidden_dim)
        self.value_linear = nn.Linear(hidden_dim, hidden_dim)
        self.scale = math.sqrt(self.head_dim)
        # Dropout
        self.dropout = nn.Dropout(p=dropout_rate_attn)
        # Output linear layer
        self.output_linear = nn.Linear(hidden_dim, hidden_dim)

    def forward(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, mask: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        batch_size = q.size(0)
        # Q, K, V
        query = self.query_linear(q)  # query, key, value: [batch_size, seq_length, hidden_dim]
        key = self.key_linear(k)
        value = self.value_linear(v)
        # Split into heads
        # [batch, len, head_num, head_dim] -> [batch, head_num, len, head_dim]
        query = query.view(batch_size, -1, self.head_num, self.head_dim).permute(0, 2, 1, 3)
        key = key.view(batch_size, -1, self.head_num, self.head_dim).permute(0, 2, 1, 3)
        value = value.view(batch_size, -1, self.head_num, self.head_dim).permute(0, 2, 1, 3)
        # Compute attention scores
        # scores: [batch, head_num, query_len, key_len]
        scores = torch.matmul(query, key.permute(0, 1, 3, 2)) / self.scale
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)
        # Compute attention weights (probabilities for each word)
        # attention: [batch, head_num, query_len, key_len]
        attention = F.softmax(scores, dim=-1)
        attention = self.dropout(attention)
        # Compute output using attention and value
        # attention_seq: [batch, head_num, query_len, head_dim]
        attention_seq = torch.matmul(attention, value).contiguous()  # Make it contiguous for view operation
        # attention_seq: [batch, query_len, hidden_dim]
        attention_seq = attention_seq.view(batch_size, -1, self.hidden_dim)
        attention_seq = self.output_linear(attention_seq)

        # Attention can be used for visualization and analysis
        return attention_seq, attention


class SublayerConnection(nn.Module):
    """
    Connects the current layer with a sublayer.
    """
    def __init__(self, hidden_dim: int, dropout_rate: float = 0.1):
        super(SublayerConnection, self).__init__()
        self.layer_norm = nn.LayerNorm(hidden_dim, eps=1e-6)
        self.dropout = nn.Dropout(p=dropout_rate)

    def forward(self, layer: torch.Tensor, sublayer: torch.Tensor) -> torch.Tensor:
        "Apply residual connection to any sublayer with the same size."
        return layer + self.dropout(self.layer_norm(sublayer))


class PositionwiseFeedForward(nn.Module):
    "Feed-Forward Network (FFN)"
    def __init__(self, hidden_dim: int, ff_dim: int, dropout_rate: float = 0.1):
        super(PositionwiseFeedForward, self).__init__()
        self.feed_forward_1 = nn.Linear(hidden_dim, ff_dim)
        self.feed_forward_2 = nn.Linear(ff_dim, hidden_dim)
        self.dropout = nn.Dropout(p=dropout_rate)
        self.activation = nn.GELU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.feed_forward_2(
            self.dropout(self.activation(self.feed_forward_1(x)))
        )


class TransformerEncoder(nn.Module):
    """
    Bidirectional Encoder = Transformer (self-attention)
    Transformer = MultiHead_Attention + Feed_Forward with sublayer connection
    """

    def __init__(
        self, hidden_dim: int, head_num: int, ff_dim: int, dropout_rate: float = 0.1, dropout_rate_attn: float = 0.1
    ):
        """
        Initializes a TransformerEncoder.

        :param hidden_dim: Hidden dimension of the transformer
        :param head_num: Number of heads in multi-head attention
        :param ff_dim: Dimension of the feed-forward hidden layer (usually 4 times hidden_dim)
        :param dropout_rate: Dropout rate
        :param dropout_rate_attn: Dropout rate for the attention layer
        """
        super(TransformerEncoder, self).__init__()
        # Initialization
        self.hidden_dim = hidden_dim
        self.head_num = head_num
        self.ff_dim = ff_dim
        self.dropout_rate = dropout_rate
        self.dropout_rate_attn = dropout_rate_attn
        # Multi-head attention
        self.attention = MultiHeadedAttention(
            head_num=self.head_num,
            hidden_dim=self.hidden_dim,
            dropout_rate_attn=self.dropout_rate_attn,
        )
        # Sublayer connection - 1 (input embeddings + input embeddings attn)
        self.input_sublayer = SublayerConnection(
            hidden_dim=self.hidden_dim, dropout_rate=self.dropout_rate
        )
        # Feed-Forward Network (FFN)
        self.feed_forward = PositionwiseFeedForward(
            hidden_dim=self.hidden_dim,
            ff_dim=self.ff_dim,
            dropout_rate=self.dropout_rate,
        )
        # Sublayer connection - 2 (sublayer connection 1's output + Feed Forward)
        self.output_sublayer = SublayerConnection(
            hidden_dim=self.hidden_dim, dropout_rate=self.dropout_rate
        )
        self.dropout = nn.Dropout(p=dropout_rate)

    def forward(self, seq: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        attention_seq, _ = self.attention(q=seq, k=seq, v=seq, mask=mask)  # _ is used to discard attention scores
        # Sublayer connection 1: Combining sequence embeddings and attention output
        connected_layer = self.input_sublayer(seq, attention_seq)
        # Sublayer connection 2: Combining connected_layer and the result of the Feed-Forward Network
        connected_layer = self.output_sublayer(connected_layer, self.feed_forward(connected_layer))

        return self.dropout(connected_layer)


class BERT(nn.Module):
    """
    BERT model: Bidirectional Encoder Representations from Transformers.
    """

    def __init__(
        self,
        vocab_size: int = 30522,
        max_len: int = 512,
        hidden_dim: int = 768,
        encoder_num: int = 12,
        head_num: int = 12,
        dropout_rate: float = 0.1,
        dropout_rate_attn: float = 0.1,
        initializer_range: float = 0.02
    ):
        """
        Initializes a BERT model.

        :param vocab_size: Vocabulary size of total words
        :param max_len: Maximum length of the input sequence
        :param hidden_dim: Hidden size of the BERT model
        :param encoder_num: Number of Transformer encoders
        :param head_num: Number of attention heads
        :param dropout_rate: Dropout rate
        :param dropout_rate_attn: Dropout rate for the attention layer
        :param initializer_range: Weight initializer range
        """
        super(BERT, self).__init__()
        self.vocab_size = vocab_size  # The number of items in the vocabulary, including [PAD] and [MASK] tokens (vocab_size + 2)
        self.max_len = max_len
        self.hidden_dim = hidden_dim
        self.encoder_num = encoder_num
        self.head_num = head_num
        self.dropout_rate = dropout_rate
        self.dropout_rate_attn = dropout_rate_attn
        self.ff_dim = hidden_dim * 4
        # Embedding
        self.embedding = BERTEmbeddings(vocab_size=self.vocab_size, embed_size=self.hidden_dim, max_len=self.max_len)
        # Transformer Encoders
        self.transformer_encoders = nn.ModuleList(
            [
                TransformerEncoder(
                    hidden_dim=self.hidden_dim,
                    head_num=self.head_num,
                    ff_dim=self.ff_dim,
                    dropout_rate=self.dropout_rate,
                    dropout_rate_attn=self.dropout_rate_attn,
                )
                for _ in range(self.encoder_num)
            ]
        )
        # Weight initialization
        self.initializer_range = initializer_range
        self.apply(self._init_weights)

    def _init_weights(self, module: nn.Module) -> None:
        """Initialize the weights."""
        if isinstance(module, (nn.Linear, nn.Embedding)):
            # Slightly different from the TF version which uses truncated_normal for initialization
            # cf https://github.com/pytorch/pytorch/pull/5617
            module.weight.data.normal_(mean=0.0, std=self.initializer_range)
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
        if isinstance(module, nn.Linear) and module.bias is not None:
            module.bias.data.zero_()

    def forward(self, seq: torch.Tensor, segment_info: Optional[torch.Tensor] = None) -> torch.Tensor:
        # Mask: [batch_size, seq_len] -> [batch_size, 1, seq_len] -> [batch_size, 1, 1, seq_len]
        # Broadcasting is used
        mask = (seq > 0).unsqueeze(1).unsqueeze(1)
        seq = self.embedding(seq, segment_info)
        for transformer in self.transformer_encoders:
            seq = transformer(seq, mask)

        return seq
