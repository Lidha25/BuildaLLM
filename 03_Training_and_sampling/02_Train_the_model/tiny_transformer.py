"""A causal Transformer whose vocabulary contains only observed tokens."""

import codecs

import torch
from torch import nn


class TinyTransformer(nn.Module):
    def __init__(self, vocab_size, context_length=48, width=64, heads=4, layers=2):
        super().__init__()
        self.context_length = context_length
        self.tokens = nn.Embedding(vocab_size, width)
        self.positions = nn.Embedding(context_length, width)
        block = nn.TransformerEncoderLayer(
            d_model=width, nhead=heads, dim_feedforward=2 * width,
            dropout=0.0, batch_first=True, norm_first=True, activation="gelu",
        )
        self.blocks = nn.TransformerEncoder(block, layers, enable_nested_tensor=False)
        self.output = nn.Sequential(nn.LayerNorm(width), nn.Linear(width, vocab_size))
        self.register_buffer(
            "causal_mask",
            torch.ones(context_length, context_length, dtype=torch.bool).triu(1),
            persistent=False,
        )

    def forward(self, ids):
        length = ids.shape[1]
        positions = torch.arange(length, device=ids.device)
        hidden = self.tokens(ids) + self.positions(positions)
        return self.output(self.blocks(hidden, mask=self.causal_mask[:length, :length]))


@torch.inference_mode()
def stream_sample(model, prompt_ids, original_ids, encoding, seed, count=24):
    """Yield readable text as tokens arrive, allowing split UTF-8 characters."""
    model.eval()
    rng = torch.Generator().manual_seed(seed)
    decoder = codecs.getincrementaldecoder("utf-8")(errors="replace")
    ids = torch.tensor([prompt_ids], dtype=torch.long)
    for _ in range(count):
        logits = model(ids[:, -model.context_length:])[0, -1]
        values, indices = logits.topk(min(30, logits.numel()))
        choice = torch.multinomial(values.softmax(0), 1, generator=rng)
        next_id = indices[choice]
        ids = torch.cat((ids, next_id.view(1, 1)), dim=1)
        yield decoder.decode(encoding.decode_single_token_bytes(original_ids[next_id.item()]))
    yield decoder.decode(b"", final=True)
