"""own_spec dual encoder (own implementation)."""
import math, torch
from torch import nn
from torch.nn import functional as F

def fourier(x, n=16, lo=0.01, hi=2000.0):
    f = torch.exp(torch.linspace(math.log(2 * math.pi / hi), math.log(2 * math.pi / lo), n, device=x.device))
    a = x.unsqueeze(-1) * f
    return torch.cat([a.sin(), a.cos()], -1)

class OwnSpec(nn.Module):
    def __init__(self, d=384, layers=6, heads=8, emb=256, fp_bits=4096, drop=0.1):
        super().__init__()
        self.cfg = dict(d=d, layers=layers, heads=heads, emb=emb, fp_bits=fp_bits, drop=drop)
        self.tok = nn.Sequential(nn.Linear(16 * 2 * 2 + 2, d), nn.GELU(), nn.Linear(d, d))
        self.add = nn.Embedding(16, d); self.ins = nn.Embedding(8, d)
        self.met = nn.Sequential(nn.Linear(3 + 32, d), nn.GELU(), nn.Linear(d, d))
        self.cls = nn.Parameter(torch.zeros(1, 1, d))
        L = nn.TransformerEncoderLayer(d, heads, 4 * d, drop, batch_first=True, norm_first=True, activation='gelu')
        self.enc = nn.TransformerEncoder(L, layers, enable_nested_tensor=False)
        self.ln = nn.LayerNorm(d)
        self.zs = nn.Linear(d, emb); self.fph = nn.Linear(d, fp_bits)
        self.mol = nn.Sequential(nn.Linear(fp_bits, 1024), nn.GELU(), nn.Dropout(drop), nn.Linear(1024, 1024), nn.GELU(), nn.Linear(1024, emb))
        self.logit_scale = nn.Parameter(torch.tensor(math.log(1 / 0.07)))

    def spec(self, x, n, m, a, ins):
        it = x[..., 1:2]
        t = torch.cat([fourier(x[..., 0]), fourier(x[..., 2].clamp(min=0)), it.sqrt(), torch.log1p(100 * it) / math.log(101)], -1)
        t = self.tok(t)
        c = self.cls.expand(len(x), -1, -1) + self.add(a)[:, None] + self.ins(ins)[:, None] + \
            self.met(torch.cat([m, fourier(m[:, 0] * 1000)], -1))[:, None]
        h = torch.cat([c, t], 1)
        pad = torch.arange(x.shape[1] + 1, device=x.device)[None] > n[:, None]
        h = self.ln(self.enc(h, src_key_padding_mask=pad)[:, 0])
        return F.normalize(self.zs(h), dim=-1), self.fph(h)

    def molecule(self, fp):  # fp: uint8 counts
        return F.normalize(self.mol(torch.log1p(fp.float())), dim=-1)

    def score(self, z, fplog, M, fp, fp_w=1.0):
        """z (S,E), fplog (S,B) logits, M (C,E), fp (C,B) counts -> (S,C) score."""
        s = z @ M.T * self.logit_scale.exp()
        b = (fp > 0).float()
        ll = F.logsigmoid(fplog) @ b.T + F.logsigmoid(-fplog) @ (1 - b).T
        return s + fp_w * ll / b.shape[1] * 100.0
