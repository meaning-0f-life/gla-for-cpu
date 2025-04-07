import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, repeat
from typing import Optional, Dict, Tuple


class GatedLinearAttention(nn.Module):
    r"""
    Реализация слоя Gated Linear Attention (GLA) из статьи:
    "Gated Linear Attention Transformers with Hardware-Efficient Training".

    Args:
        hidden_size (int): Размер скрытого слоя. По умолчанию: 1024.
        expand_k (float): Коэффициент расширения для размерности ключа. По умолчанию: 0.5.
        expand_v (float): Коэффициент расширения для размерности значения. По умолчанию: 1.0.
        num_heads (int): Количество голов внимания. По умолчанию: 4.
        num_kv_heads (int): Количество голов для ключа и значения (используется для MQA). По умолчанию: None.
        use_short_conv (bool): Использовать ли короткие свертки. По умолчанию: False.
        conv_size (int): Размер ядра свертки. По умолчанию: 4.
        use_output_gate (bool): Использовать ли выходной gate. По умолчанию: True.
        gate_fn (str): Функция активации для gate. По умолчанию: 'swish'.
        norm_eps (float): Эпсилон для LayerNorm. По умолчанию: 1e-5.
        gate_logit_normalizer (int): Нормализатор для gate logits. По умолчанию: 16.
        gate_low_rank_dim (int): Низкоранговая размерность для gate. По умолчанию: 16.
        clamp_min (float): Минимальное значение для gate logits. По умолчанию: None.
        fuse_norm (bool): Объединять ли нормировку и gate. По умолчанию: True.
        layer_idx (int): Индекс слоя. По умолчанию: None.
        chunk_size (bool): Число чанков. По умолчанию: 64.
        subchunk_size (int): Число подчанков. По умолчанию: 16.
    """

    def __init__(
        self,
        hidden_size: int = 1024,
        expand_k: float = 0.5,
        expand_v: float = 1.0,
        num_heads: int = 4,
        num_kv_heads: Optional[int] = None,
        use_short_conv: bool = False,
        conv_size: int = 4,
        use_output_gate: bool = True,
        gate_fn: str = 'swish',
        norm_eps: float = 1e-5,
        gate_logit_normalizer: int = 16,
        gate_low_rank_dim: int = 16,
        clamp_min: Optional[float] = None,
        fuse_norm: bool = True,
        layer_idx: int = None,
        chunk_size: int = 64,
        subchunk_size: int = 16,
    ) -> "GatedLinearAttention":
        super().__init__()

        self.hidden_size = hidden_size
        self.expand_k = expand_k
        self.expand_v = expand_v
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads if num_kv_heads is not None else num_heads
        self.num_kv_groups = self.num_heads // self.num_kv_heads

        self.chunk_size = chunk_size
        self.subchunk_size = subchunk_size

        self.use_short_conv = use_short_conv
        self.conv_size = conv_size
        self.use_output_gate = use_output_gate

        self.key_dim = int(hidden_size * expand_k)
        self.value_dim = int(hidden_size * expand_v)
        self.key_dim_per_group = self.key_dim // self.num_kv_groups
        self.value_dim_per_group = self.value_dim // self.num_kv_groups
        self.clamp_min = clamp_min
        self.layer_idx = layer_idx

        assert self.key_dim % num_heads == 0, f"Размерность ключа должна делиться на количество голов {num_heads}"
        assert self.value_dim % num_heads == 0, f"Размерность значения должна делиться на количество голов {num_heads}"

        self.head_k_dim = self.key_dim // num_heads
        self.head_v_dim = self.value_dim // num_heads

        # Линейные проекции для Q, K, V
        self.q_proj = nn.Linear(hidden_size, self.key_dim, bias=False)
        self.k_proj = nn.Linear(hidden_size, self.key_dim_per_group, bias=False)
        self.v_proj = nn.Linear(hidden_size, self.value_dim_per_group, bias=False)
        if self.use_output_gate:
            self.g_proj = nn.Linear(hidden_size, self.value_dim, bias=False)

        # Проекция для gate
        self.gk_proj = nn.Sequential(
            nn.Linear(hidden_size, gate_low_rank_dim, bias=False),
            nn.Linear(gate_low_rank_dim, self.key_dim_per_group, bias=True)
        )
        self.o_proj = nn.Linear(self.value_dim, hidden_size, bias=False)

        # Нормализация и gate
        if gate_fn == 'swish' and fuse_norm and use_output_gate:
            self.g_norm_swish_gate = nn.Sequential(
                nn.LayerNorm(self.head_v_dim, eps=norm_eps),
                nn.SiLU()  # Swish активация
            )
            self.fuse_norm_and_gate = True
        else:
            self.fuse_norm_and_gate = False
            self.g_norm = nn.LayerNorm(self.head_v_dim, eps=norm_eps)
            self.gate_fn = nn.SiLU() if gate_fn == 'swish' else nn.Identity()

        self.gate_logit_normalizer = gate_logit_normalizer

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        past_key_values: Optional[Dict] = None,
        use_cache: Optional[bool] = False,
        **kwargs
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Dict]]:
        # Если attention_mask предоставлен, проверяем его формат
        if attention_mask is not None:
            assert len(attention_mask.shape) == 2, (
                "Ожидается attention_mask как матрица 0-1 с размерностью [batch_size, seq_len]."
            )

        # Линейные проекции для Q, K, V
        q = self.q_proj(hidden_states) # [batch_size, seq_len, key_dim]
        k = self.k_proj(hidden_states) # [batch_size, seq_len, key_dim_per_group]
        v = self.v_proj(hidden_states) # [batch_size, seq_len, value_dim_per_group]
        gk = self.gk_proj(hidden_states) # [batch_size, seq_len, key_dim_per_group]

        # Применение маски, если она есть
        if attention_mask is not None: # attention_mask = [batch_size, seq_len] (матрица 0 и 1)
            v = v * attention_mask[:, -v.shape[-2]:, None] # v.shape[-2] — это seq_len
            # Действует на v[i, j, :]

        # Перегруппировка Q, K, V для многоголового внимания
        q = rearrange(q, 'b t (h d) -> b t h d', d=self.head_k_dim) # [batch_size, seq_len, num_heads, head_dim]
        if self.num_kv_groups > 1:
            k, gk = (repeat(x, 'b t (h d) -> b t (h g) d', g=self.num_kv_groups, d=self.head_k_dim) for x in (k, gk))
            v = repeat(v, 'b t (h d) -> b t (h g) d', g=self.num_kv_groups, d=self.head_v_dim)
        else:
            k, gk = (rearrange(x, 'b t (h d) -> b t h d', d=self.head_k_dim) for x in (k, gk))
            v = rearrange(v, 'b t (h d) -> b t h d', d=self.head_v_dim)

        # Применение gate
        gk = F.logsigmoid(gk) / self.gate_logit_normalizer
        if self.clamp_min is not None:
            gk = torch.clamp_min(gk, self.clamp_min)

        # Линейное внимание с чанкованием
        o = self._chunk_gla(q, k, v, gk)

        # Применение выходного gate
        if self.use_output_gate:
            g = self.g_proj(hidden_states)
            if self.fuse_norm_and_gate:
                g = rearrange(g, 'b t (h d) -> b t h d', d=self.head_v_dim)
                o = self.g_norm_swish_gate(o + g)
                o = rearrange(o, 'b t h d -> b t (h d)')
            else:
                o = rearrange(self.g_norm(o), 'b t h d -> b t (h d)')
                o = o * self.gate_fn(g)
        else:
            o = rearrange(self.g_norm(o), 'b t h d -> b t (h d)')

        # Финальная проекция
        o = self.o_proj(o)

        return o, None, past_key_values

    def _chunk_gla(self, q, k, v, gk):
        """
        Реализация Gated Linear Attention с чанкованием.

        Args:
            q (torch.Tensor): Тензор запросов (query) с размерностью [batch_size, seq_len, num_heads, head_dim].
            k (torch.Tensor): Тензор ключей (key) с размерностью [batch_size, seq_len, num_heads, head_dim].
            v (torch.Tensor): Тензор значений (value) с размерностью [batch_size, seq_len, num_heads, head_v_dim].
            gk (torch.Tensor): Тензор ворот (gate) с размерностью [batch_size, seq_len, num_heads, head_dim].

        Returns:
            torch.Tensor: Выходной тензор с размерностью [batch_size, seq_len, num_heads, head_v_dim].
        """
        batch_size, seq_len, num_heads, head_dim = q.shape
        head_v_dim = v.shape[-1]  # Размерность значений (head_v_dim)
        chunk_size = self.chunk_size
        subchunk_size = self.subchunk_size

        # Инициализация скрытого состояния
        S = torch.zeros(batch_size, num_heads, head_dim, head_v_dim, device=q.device)
        O = torch.zeros_like(v)

        # Разделение последовательности на чанки
        for i in range(0, seq_len // chunk_size):
            # Индексы для текущего чанка
            chunk_start = i * chunk_size
            chunk_end = (i + 1) * chunk_size
            r = range(chunk_start, chunk_end)

            # Получение чанков для Q, K, V, G
            bq = q[:, r, :, :]
            bk = k[:, r, :, :]
            bv = v[:, r, :, :]
            bg = gk[:, r, :, :]

            # Вычисление кумулятивной суммы для ворот внутри чанка
            B = torch.cumsum(bg, dim=1)

            # Межчанковое внимание с использованием матричного умножения
            q_exp = bq * torch.exp(B[:, -1, :, :].unsqueeze(1))
            k_exp = bk * torch.exp(B[:, -1, :, :].unsqueeze(1) - B)
            g_exp = torch.exp(B[:, -1, :, :].unsqueeze(1))

            # Обновление скрытого состояния
            S = torch.einsum('bnhd,bhde->bhde', g_exp, S) + torch.einsum('bnhd,bnhe->bhde', k_exp, bv)
            O[:, r, :, :] = torch.einsum('bnhd,bhde->bnhe', q_exp, S)

            # Внутричанковое внимание (вторичное чанкование)
            for j in range(0, chunk_size // subchunk_size):
                # Индексы для текущего подчанка
                subchunk_start = j * subchunk_size
                subchunk_end = (j + 1) * subchunk_size
                t = range(subchunk_start, subchunk_end)

                # Получение подчанков для Q, K, V, G
                sq = bq[:, t, :, :]
                sk = bk[:, t, :, :]
                sv = bv[:, t, :, :]
                sb = B[:, t, :, :]

                # Внутриподчанковое внимание без матричного умножения
                p = torch.zeros(batch_size, subchunk_size, subchunk_size, num_heads, head_v_dim, device=q.device)
                for m in range(subchunk_size):
                    for n in range(m + 1):
                        # Вычисление внимания с сохранением размерности head_v_dim
                        attention = sq[:, m, :, :] * sk[:, n, :, :] * torch.exp(sb[:, m, :, :] - sb[:, n, :, :])
                        p[:, m, n, :, :] = attention.sum(dim=-1, keepdim=True)  # Сохраняем размерность head_v_dim

                # Обновление выхода для подчанка
                for idx in t:
                    # Используем правильный срез для p и sv
                    p_slice = p[:, :, idx - subchunk_start, :, :]  # [batch_size, subchunk_size, num_heads, head_v_dim]
                    sv_slice = sv[:, idx - subchunk_start, :, :]  # [batch_size, num_heads, head_v_dim]

                    # Перестановка осей в p_slice
                    p_slice = p_slice.permute(0, 2, 3, 1)  # [batch_size, num_heads, head_v_dim, subchunk_size]

                    # Обновление выхода
                    O[:, chunk_start + idx, :, :] += torch.einsum('bnhd,bnh->bnh', p_slice, sv_slice)

        return O