import torch
import torch.nn as nn
from gla import GatedLinearAttention


class GLATransformer(nn.Module):
    def __init__(self, d_model, n_heads, d_k, d_v, num_layers, vocab_size):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model) # [batch_size, seq_len, d_model]
        self.layers = nn.ModuleList([
            GatedLinearAttention(
                hidden_size=d_model,
                expand_k=0.5,
                expand_v=1.0,
                num_heads=n_heads,
                num_kv_heads=n_heads,
                use_output_gate=True,
                gate_fn='swish',
                norm_eps=1e-5,
                gate_logit_normalizer=16,
                gate_low_rank_dim=16,
                fuse_norm=True,
                layer_idx=i,
                chunk_size=64,
                subchunk_size=16
            ) for i in range(num_layers)
        ])
        self.norm = nn.LayerNorm(d_model)
        self.lm_head = nn.Linear(d_model, vocab_size)

    def forward(self, input_ids, attention_mask=None):
        assert torch.max(input_ids) < self.embedding.num_embeddings, f"Max index {torch.max(input_ids)} out of range!"
        hidden_states = self.embedding(input_ids)
        for layer in self.layers:
            hidden_states, _, _ = layer(hidden_states, attention_mask)
        hidden_states = self.norm(hidden_states)
        logits = self.lm_head(hidden_states)
        return logits