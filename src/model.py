import torch 
import torch.nn as nn
import torch.nn.functional as F
from encoder import TransformerEncoderBlock
from decoder import TransformerDecoderBlock
from Utils.positional_encoding import PositionalEncoding
class Transformer(nn.Module):
    def __init__(self, src_vocab_size, tgt_vocab_size, embed_dim=512, num_heads=8, ff_dim=2048, num_layers=6, dropout=0.1, max_len=512):
        super().__init__()

        self.src_embedding = nn.Embedding(src_vocab_size, embed_dim)
        self.tgt_embedding = nn.Embedding(tgt_vocab_size, embed_dim)
        self.pos_encoder = PositionalEncoding(embed_dim, max_len)

        self.encoder_layers = nn.ModuleList([
            TransformerEncoderBlock(embed_dim, num_heads, ff_dim, dropout)
            for _ in range(num_layers)
        ])

        self.decoder_layers = nn.ModuleList([
            TransformerDecoderBlock(embed_dim, num_heads, ff_dim, dropout)
            for _ in range(num_layers)
        ])

        self.fc_out = nn.Linear(embed_dim, tgt_vocab_size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, src, tgt, src_mask=None, tgt_mask=None):
        # Embedding + Positional Encoding
        src = self.dropout(self.pos_encoder(self.src_embedding(src)))  # (B, T_src, C)
        tgt = self.dropout(self.pos_encoder(self.tgt_embedding(tgt)))  # (B, T_tgt, C)

        # Encoder
        for layer in self.encoder_layers:
            src = layer(src, src_mask)

        enc_output = src

        # Decoder
        for layer in self.decoder_layers:
            tgt = layer(tgt, enc_output, tgt_mask, src_mask)

        # Final projection to vocab
        out = self.fc_out(tgt)  # (B, T, vocab_size)

        return out
