import torch 
import torch.nn as nn
import torch.nn.functional as F
from encoder import TransformerEncoderBlock
from decoder import TransformerDecoderBlock
from Utils.positional_encoding import PositionalEncoding


class Transformer(nn.Module):
    def __init__(self, src_vocab_size, tgt_vocab_size, embed_dim=64, num_heads=4, ff_dim=256, num_layers=4, dropout=0.0, max_len=32):
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
        src = self.dropout(self.pos_encoder(self.src_embedding(src)))  # (B, T_src, C)
        tgt = self.dropout(self.pos_encoder(self.tgt_embedding(tgt)))  # (B, T_tgt, C)
        for layer in self.encoder_layers:
            src = layer(src, src_mask)
        enc_output = src
        for layer in self.decoder_layers:
            tgt = layer(tgt, enc_output, tgt_mask, src_mask)
        out = self.fc_out(tgt)  # (B, T, vocab_size)
        return out

    def generate(self, idx, max_new_tokens, block_size=32):
        device = idx.device
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -block_size:]  # Crop to block_size
            mask = torch.tril(torch.ones(idx_cond.size(1), idx_cond.size(1))).to(device)
            # Encode the context
            src_emb = self.dropout(self.pos_encoder(self.src_embedding(idx_cond)))
            enc_output = src_emb
            for layer in self.encoder_layers:
                enc_output = layer(enc_output)
            # Decode one step
            tgt_emb = self.dropout(self.pos_encoder(self.tgt_embedding(idx_cond)))
            for layer in self.decoder_layers:
                tgt_emb = layer(tgt_emb, enc_output, tgt_mask=mask)
            logits = self.fc_out(tgt_emb)[:, -1, :]  # Last time step
            probs = torch.nn.functional.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, idx_next), dim=1)
        return idx
