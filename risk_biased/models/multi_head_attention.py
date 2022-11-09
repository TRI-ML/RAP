# Implementation from https://einops.rocks/pytorch-examples.html slightly changed


import math

from typing import Tuple
import torch
from torch import nn
from einops import rearrange, repeat


class MultiHeadAttention(nn.Module):
    """
    This is a slightly modified version of the original implementation from https://einops.rocks/pytorch-examples.html of multihead attention.
    It keeps the original dimension division per head and masks the attention matrix before and after the softmax to support full row masking.

    Args:
        d_model: the input feature dimension of the model
        n_head: the number of heads in the multihead attention
        d_k: the dimension of the key and query in the multihead attention
        d_v: the dimension of the value in the multihead attention
    """

    def __init__(self, d_model: int, n_head: int, d_k: torch.Tensor, d_v: torch.Tensor):
        super().__init__()
        self.n_head = n_head

        self.w_qs = nn.Linear(d_model, int(d_k / n_head) * n_head)
        self.w_ks = nn.Linear(d_model, int(d_k / n_head) * n_head)
        self.w_vs = nn.Linear(d_model, int(d_v / n_head) * n_head)
        self.w_rs = nn.Linear(d_model, int(d_v / n_head) * n_head)

        nn.init.normal_(self.w_qs.weight, mean=0, std=math.sqrt(2.0 / (d_model + d_k)))
        nn.init.normal_(self.w_ks.weight, mean=0, std=math.sqrt(2.0 / (d_model + d_k)))
        nn.init.normal_(self.w_vs.weight, mean=0, std=math.sqrt(2.0 / (d_model + d_v)))
        nn.init.normal_(self.w_rs.weight, mean=0, std=math.sqrt(2.0 / (d_model + d_v)))

        self.fc = nn.Linear(int(d_v / n_head) * n_head, d_model)
        nn.init.xavier_normal_(self.fc.weight)
        self.layer_norm = nn.LayerNorm(d_model)

    def forward(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        mask: torch.Tensor = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute the masked multi-head attention given the query, key and value tensors.

        Args:
            q: the query tensor of shape [batch_size, number_of_agents, d_model]
            k: the key tensor of shape [batch_size, number_of_objects, d_model]
            v: the value tensor of shape [batch_size, number_of_objects, d_model]
            mask: the mask tensor of shape [batch_size, number_of_agents, number_of_objects]

        Returns:
            [
                The attention output tensor of shape [batch_size, number_of_agents, d_model],
                The attention matrix of shape [batch_size, number_of_agents, number_of_objects]
            ]
        """
        residual = q.clone()
        r = self.w_rs(q)
        q = rearrange(self.w_qs(q), "b a (head k) -> head b a k", head=self.n_head)
        k = rearrange(self.w_ks(k), "b o (head k) -> head b o k", head=self.n_head)
        v = rearrange(self.w_vs(v), "b o (head v) -> head b o v", head=self.n_head)
        attn = torch.einsum("hbak,hbok->hbao", [q, k]) / math.sqrt(q.shape[-1])
        if mask is not None:
            # b: batch, a: agent, o: object, h: head
            mask = repeat(mask, "b a o -> h b a o", h=self.n_head)
            attn = attn.masked_fill(mask == 0, -math.inf)
        attn = torch.softmax(attn, dim=3)
        # Here we need to mask again because some lines might be all -inf in the softmax which gives Nan...
        attn = attn.masked_fill(mask == 0, 0)
        output = torch.einsum("hbao,hbov->hbav", [attn, v])
        output = rearrange(output, "head b a v -> b a (head v)")
        output = self.fc(output * r)
        output = self.layer_norm(output + residual)
        return output, attn
