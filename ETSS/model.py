import torch
import torch.nn as nn
import torch.nn.functional as F


class SurrogateSpike(torch.autograd.Function):
    @staticmethod
    def forward(ctx, v, v_th, sg_beta):
        s = (v >= v_th).float()
        ctx.save_for_backward(v - v_th)
        ctx.sg_beta = sg_beta
        return s

    @staticmethod
    def backward(ctx, grad_output):
        (z,) = ctx.saved_tensors
        beta = ctx.sg_beta
        sigma = torch.sigmoid(beta * z)
        grad_v = grad_output * beta * sigma * (1.0 - sigma)
        return grad_v, None, None


class LIF(nn.Module):
    def __init__(
        self,
        d_model,
        tau_init=2.0,
        dt=1.0,
        reset_scale=0.5,
        sg_beta=5.0,
        learn_tau=True,
    ):
        super().__init__()

        self.dt = dt
        self.reset_scale = reset_scale
        self.sg_beta = sg_beta
        self.v_th = nn.Parameter(torch.ones(1, 1, 1, d_model))

        if learn_tau:
            self.tau_param = nn.Parameter(torch.full((1, 1, 1, d_model), tau_init))
        else:
            self.tau_param = None
            self.register_buffer("tau_const", torch.tensor(tau_init, dtype=torch.float32))

    def alpha(self):
        if self.tau_param is not None:
            tau = F.softplus(self.tau_param)
        else:
            tau = self.tau_const.to(device=self.v_th.device, dtype=self.v_th.dtype)

        return torch.exp(-self.dt / tau)

    def forward(self, x):
        B, C, T, D = x.shape

        v_post = torch.zeros(B, C, 1, D, device=x.device, dtype=x.dtype)
        alpha = self.alpha().to(device=x.device, dtype=x.dtype)
        v_th = self.v_th.to(device=x.device, dtype=x.dtype)

        spikes = []
        pre_vs = []
        post_vs = []

        for t in range(T):
            x_t = x[:, :, t:t + 1, :]

            v_pre = alpha * v_post + x_t
            s = SurrogateSpike.apply(v_pre, v_th, self.sg_beta)
            v_post = v_pre - s * v_th * self.reset_scale

            spikes.append(s)
            pre_vs.append(v_pre)
            post_vs.append(v_post)

        s_all = torch.cat(spikes, dim=2)
        v_pre_all = torch.cat(pre_vs, dim=2)
        v_post_all = torch.cat(post_vs, dim=2)

        return s_all, v_pre_all, v_post_all


class FixedWindowLIF(nn.Module):
    def __init__(
        self,
        d_model,
        window_size=16,
        tau_init=2.0,
        dt=1.0,
        reset_scale=0.5,
        sg_beta=5.0,
        learn_tau=True,
    ):
        super().__init__()

        self.window_size = window_size
        self.lif = LIF(
            d_model=d_model,
            tau_init=tau_init,
            dt=dt,
            reset_scale=reset_scale,
            sg_beta=sg_beta,
            learn_tau=learn_tau,
        )

    def forward(self, x):
        B, C, T, D = x.shape
        W = self.window_size

        if T <= 0:
            raise ValueError("Input sequence length T must be positive.")

        pad_len = (W - T % W) % W

        if pad_len > 0:
            pad = torch.zeros(B, C, pad_len, D, device=x.device, dtype=x.dtype)
            x_pad = torch.cat([x, pad], dim=2)
        else:
            x_pad = x

        T_pad = x_pad.shape[2]
        num_windows = T_pad // W

        x_seg = x_pad.reshape(B, C, num_windows, W, D)
        x_seg = x_seg.permute(0, 2, 1, 3, 4).contiguous()
        x_seg = x_seg.reshape(B * num_windows, C, W, D)

        s_seg, v_pre_seg, v_post_seg = self.lif(x_seg)

        s_all = s_seg.reshape(B, num_windows, C, W, D)
        s_all = s_all.permute(0, 2, 1, 3, 4).contiguous()
        s_all = s_all.reshape(B, C, T_pad, D)

        v_pre_all = v_pre_seg.reshape(B, num_windows, C, W, D)
        v_pre_all = v_pre_all.permute(0, 2, 1, 3, 4).contiguous()
        v_pre_all = v_pre_all.reshape(B, C, T_pad, D)

        v_post_all = v_post_seg.reshape(B, num_windows, C, W, D)
        v_post_all = v_post_all.permute(0, 2, 1, 3, 4).contiguous()
        v_post_all = v_post_all.reshape(B, C, T_pad, D)

        return s_all[:, :, :T, :], v_pre_all[:, :, :T, :], v_post_all[:, :, :T, :]


class DVSProcessor(nn.Module):
    def __init__(self, d_model=256, input_dim=256):
        super().__init__()

        self.d_model = d_model
        self.input_dim = input_dim

        if input_dim != d_model:
            self.dvs_proj = nn.Linear(input_dim, d_model, bias=False)
        else:
            self.dvs_proj = None

        self.dvs_ln = nn.LayerNorm(d_model)

    def forward(self, raw_dvs_input):
        B, C, T, D_in = raw_dvs_input.shape

        if D_in != self.input_dim:
            raise ValueError(f"DVSProcessor expected input_dim={self.input_dim}, got {D_in}")

        x = raw_dvs_input

        if self.dvs_proj is not None:
            x = x.reshape(B * C * T, D_in)
            x = self.dvs_proj(x)
            x = x.reshape(B, C, T, self.d_model)

        return self.dvs_ln(x)


class RGBProcessor(nn.Module):
    def __init__(self, d_model=256, input_dim=2048):
        super().__init__()

        self.d_model = d_model
        self.input_dim = input_dim
        self.rgb_proj = nn.Linear(input_dim, d_model, bias=False)
        self.rgb_ln = nn.LayerNorm(d_model)

    def forward(self, raw_rgb_input):
        B, C, T, D_in = raw_rgb_input.shape

        if D_in != self.input_dim:
            raise ValueError(f"RGBProcessor expected input_dim={self.input_dim}, got {D_in}")

        x = raw_rgb_input.reshape(B * C * T, D_in)
        x = self.rgb_proj(x)
        x = self.rgb_ln(x)
        x = x.reshape(B, C, T, self.d_model)

        return x


class SpikingGateMLP(nn.Module):
    def __init__(self, d_model, lif_window=16, sg_beta=5.0):
        super().__init__()

        self.fc1 = nn.Linear(2 * d_model, d_model)
        self.lif = FixedWindowLIF(
            d_model=d_model,
            window_size=lif_window,
            sg_beta=sg_beta,
        )
        self.fc2 = nn.Linear(d_model, d_model)

    def forward(self, gate_in):
        B, C, T, _ = gate_in.shape
        D = self.fc2.out_features

        h = self.fc1(gate_in.reshape(B * C * T, -1))
        h = h.reshape(B, C, T, D)

        s, _, _ = self.lif(h)

        gate_logits = self.fc2(s.reshape(B * C * T, D))
        gate_logits = gate_logits.reshape(B, C, T, D)

        return torch.sigmoid(gate_logits)


class MembraneFeatureInjectionFusion(nn.Module):
    def __init__(
        self,
        d_model=256,
        dropout=0.1,
        lif_window=16,
        use_spiking_gate=False,
    ):
        super().__init__()

        self.d_model = d_model
        self.use_spiking_gate = use_spiking_gate
        self.inj_proj = nn.Linear(d_model, d_model, bias=False)

        if use_spiking_gate:
            self.gate_mlp = SpikingGateMLP(
                d_model=d_model,
                lif_window=lif_window,
            )
        else:
            self.gate_mlp = nn.Sequential(
                nn.Linear(2 * d_model, d_model),
                nn.GELU(),
                nn.Linear(d_model, d_model),
                nn.Sigmoid(),
            )

        self.lif_fusion = FixedWindowLIF(
            d_model=d_model,
            window_size=lif_window,
        )

        self.out_ln = nn.LayerNorm(d_model)
        self.drop = nn.Dropout(dropout)

    def forward(self, v_dvs_pre, v_rgb_pre=None):
        B, C, T, D = v_dvs_pre.shape

        if D != self.d_model:
            raise ValueError(
                f"MembraneFeatureInjectionFusion expected d_model={self.d_model}, got {D}"
            )

        if v_rgb_pre is not None:
            if v_rgb_pre.shape != v_dvs_pre.shape:
                raise ValueError(
                    f"v_rgb_pre shape {v_rgb_pre.shape} must match v_dvs_pre shape {v_dvs_pre.shape}"
                )

            inj = self.inj_proj(v_rgb_pre.reshape(B * C * T, D)).reshape(B, C, T, D)
            gate_in = torch.cat([v_dvs_pre, v_rgb_pre], dim=-1)

            if self.use_spiking_gate:
                gate = self.gate_mlp(gate_in)
            else:
                gate = self.gate_mlp(gate_in.reshape(B * C * T, 2 * D))
                gate = gate.reshape(B, C, T, D)

            injected_membrane = v_dvs_pre + gate * inj

        else:
            gate = None
            inj = None
            injected_membrane = v_dvs_pre

        s_fused, v_fused_pre, v_fused_post = self.lif_fusion(injected_membrane)

        out = self.out_ln(v_fused_pre)
        out = self.drop(out)

        return out, {
            "gate": gate,
            "inj": inj,
            "injected_membrane": injected_membrane,
            "s_fused": s_fused,
            "v_fused_pre": v_fused_pre,
            "v_fused_post": v_fused_post,
        }


class MFIFusion(nn.Module):
    def __init__(
        self,
        F_dvs=256,
        F_rgb=2048,
        d_model=256,
        dropout=0.1,
        lif_window=16,
        use_spiking_gate=False,
    ):
        super().__init__()

        self.lif_window = lif_window

        self.dvs_processor = DVSProcessor(
            d_model=d_model,
            input_dim=F_dvs,
        )

        self.lif_dvs = FixedWindowLIF(
            d_model=d_model,
            window_size=lif_window,
        )

        self.rgb_processor = RGBProcessor(
            d_model=d_model,
            input_dim=F_rgb,
        )

        self.lif_rgb = FixedWindowLIF(
            d_model=d_model,
            window_size=lif_window,
        )

        self.mfi_fusion = MembraneFeatureInjectionFusion(
            d_model=d_model,
            dropout=dropout,
            lif_window=lif_window,
            use_spiking_gate=use_spiking_gate,
        )

    def forward(self, inputs_dvs, inputs_rgb=None):
        B, C, T, _ = inputs_dvs.shape

        if inputs_rgb is not None:
            if inputs_rgb.shape[:3] != inputs_dvs.shape[:3]:
                raise ValueError(
                    f"inputs_rgb shape {inputs_rgb.shape} must have same [B, C, T] as inputs_dvs {inputs_dvs.shape}"
                )

        x_dvs = self.dvs_processor(inputs_dvs)
        s_dvs, v_dvs_pre, v_dvs_post = self.lif_dvs(x_dvs)

        x_rgb = None
        s_rgb = None
        v_rgb_pre = None
        v_rgb_post = None

        if inputs_rgb is not None:
            x_rgb = self.rgb_processor(inputs_rgb)
            s_rgb, v_rgb_pre, v_rgb_post = self.lif_rgb(x_rgb)

        out, fusion_info = self.mfi_fusion(v_dvs_pre, v_rgb_pre)

        fusion_info.update({
            "lif_window": self.lif_window,
            "original_T": T,
            "x_dvs": x_dvs,
            "s_dvs": s_dvs,
            "v_dvs_pre": v_dvs_pre,
            "v_dvs_post": v_dvs_post,
            "x_rgb": x_rgb,
            "s_rgb": s_rgb,
            "v_rgb_pre": v_rgb_pre,
            "v_rgb_post": v_rgb_post,
        })

        return out, fusion_info


class TemporalHead(nn.Module):
    def __init__(
        self,
        d_model,
        out_ch=1,
        use_temporal_conv=True,
    ):
        super().__init__()

        self.use_temporal_conv = use_temporal_conv

        if use_temporal_conv:
            self.tconv = nn.Conv1d(
                in_channels=d_model,
                out_channels=d_model,
                kernel_size=3,
                padding=1,
                groups=d_model,
            )

        self.norm = nn.LayerNorm(d_model)
        self.proj = nn.Linear(d_model, out_ch)

    def forward(self, x):
        B, C, T, D = x.shape

        if self.use_temporal_conv:
            x1 = x.reshape(B * C, T, D).transpose(1, 2)
            x1 = self.tconv(x1)
            x = x1.transpose(1, 2).reshape(B, C, T, D)

        x = self.norm(x)

        return self.proj(x)


class Model(nn.Module):
    def __init__(
        self,
        F_dvs=256,
        F_rgb=2048,
        d_model=256,
        out_ch=1,
        lif_window=16,
        use_spiking_gate=False,
    ):
        super().__init__()

        self.fusion = MFIFusion(
            F_dvs=F_dvs,
            F_rgb=F_rgb,
            d_model=d_model,
            dropout=0.1,
            lif_window=lif_window,
            use_spiking_gate=use_spiking_gate,
        )

        self.head = TemporalHead(
            d_model=d_model,
            out_ch=out_ch,
            use_temporal_conv=True,
        )

    def forward(self, inputs_dvs, inputs_rgb=None):
        fused, fusion_info = self.fusion(inputs_dvs, inputs_rgb)

        logits_c = self.head(fused)
        logits = logits_c.mean(dim=1)

        return logits, fusion_info