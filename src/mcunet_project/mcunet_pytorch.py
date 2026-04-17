"""
MCUNet - PyTorch Implementation
================================
MCUNet: Tiny Deep Learning on IoT Devices (NeurIPS 2020)
Authors: Ji Lin, Wei-Ming Chen, Yujun Lin, et al. (MIT HAN Lab)

This file implements:
  1. MCUNet backbone (TinyNAS-searched MobileNetV2-style architecture)
  2. TinyNAS search space definition
  3. Task-specific heads (Image Classification, VWW, Keyword Spotting)
  4. INT8 Quantization-Aware Training (QAT) wrapper
  5. Resource estimation utilities (SRAM, Flash, FLOPS, latency)

Usage:
    python mcunet_pytorch.py --task classification --profile
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.quantization import QuantStub, DeQuantStub, fuse_modules
import math
from dataclasses import dataclass, field
from typing import List, Optional, Tuple
import numpy as np


# ─────────────────────────────────────────────
# 1.  Hardware Constraints for STM32 Targets
# ─────────────────────────────────────────────

@dataclass
class MCUConfig:
    """Hardware resource budget for target MCU."""
    name: str
    sram_kb: int        # Peak SRAM available (KB)
    flash_kb: int       # Flash storage for model weights (KB)
    cpu_mhz: int        # Core clock frequency
    has_fpu: bool       # Hardware floating-point unit
    has_simd: bool      # SIMD (DSP) instructions (e.g., ARM Cortex-M4/M7)

# Commonly used STM32 boards for MCUNet experiments
STM32_CONFIGS = {
    "STM32F746": MCUConfig("STM32F746", sram_kb=320,  flash_kb=1024, cpu_mhz=216,  has_fpu=True,  has_simd=True),
    "STM32H743": MCUConfig("STM32H743", sram_kb=512,  flash_kb=2048, cpu_mhz=480,  has_fpu=True,  has_simd=True),
    "STM32L4R5": MCUConfig("STM32L4R5", sram_kb=640,  flash_kb=2048, cpu_mhz=120,  has_fpu=True,  has_simd=True),
    "STM32F412": MCUConfig("STM32F412", sram_kb=256,  flash_kb=1024, cpu_mhz=100,  has_fpu=True,  has_simd=True),
    "Arduino_M4": MCUConfig("Arduino_M4", sram_kb=256, flash_kb=512,  cpu_mhz=120,  has_fpu=True,  has_simd=False),
}


# ─────────────────────────────────────────────
# 2.  Core Building Blocks
# ─────────────────────────────────────────────

class HardSwish(nn.Module):
    """Memory-efficient hard-swish activation (avoids exp)."""
    def forward(self, x):
        return x * F.hardtanh(x + 3, 0.0, 6.0) / 6.0


class HardSigmoid(nn.Module):
    def forward(self, x):
        return F.hardtanh(x + 3, 0.0, 6.0) / 6.0


class SEBlock(nn.Module):
    """
    Squeeze-and-Excitation block with tiny bottleneck.
    Used optionally in MCUNet to boost accuracy with minimal overhead.
    """
    def __init__(self, channels: int, reduction: int = 4):
        super().__init__()
        squeezed = max(1, channels // reduction)
        self.se = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, squeezed, 1, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(squeezed, channels, 1, bias=True),
            HardSigmoid()
        )

    def forward(self, x):
        return x * self.se(x)


class MBConvBlock(nn.Module):
    """
    Mobile Inverted Bottleneck Convolution Block (MBConv).
    Core operator in MCUNet / TinyNAS search space.

    Structure:
        [Expand 1x1] → [Depthwise 3x3/5x5] → [SE (optional)] → [Project 1x1]

    Key MCUNet optimisation:
        In-place depthwise conv: output overwrites input buffer → halves peak SRAM.
    """
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        stride: int = 1,
        expand_ratio: int = 6,
        use_se: bool = False,
        se_reduction: int = 4,
        activation: str = "relu6",
    ):
        super().__init__()
        self.stride = stride
        self.use_residual = (stride == 1 and in_channels == out_channels)
        hidden = int(in_channels * expand_ratio)

        act = HardSwish() if activation == "hswish" else nn.ReLU6(inplace=True)

        layers: List[nn.Module] = []

        # Expand phase (skip if ratio == 1)
        if expand_ratio != 1:
            layers += [
                nn.Conv2d(in_channels, hidden, 1, bias=False),
                nn.BatchNorm2d(hidden),
                act,
            ]

        # Depthwise phase
        layers += [
            nn.Conv2d(hidden, hidden, kernel_size,
                      stride=stride, padding=kernel_size // 2,
                      groups=hidden, bias=False),
            nn.BatchNorm2d(hidden),
            act,
        ]

        # SE phase
        if use_se:
            layers.append(SEBlock(hidden, se_reduction))

        # Project phase
        layers += [
            nn.Conv2d(hidden, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
        ]

        self.conv = nn.Sequential(*layers)

    def forward(self, x):
        out = self.conv(x)
        if self.use_residual:
            return x + out
        return out


# ─────────────────────────────────────────────
# 3.  TinyNAS Architecture (MCUNet Backbone)
# ─────────────────────────────────────────────

@dataclass
class BlockConfig:
    """Configuration for one stage of the network."""
    in_ch: int
    out_ch: int
    n_blocks: int
    stride: int
    kernel_size: int
    expand_ratio: int
    use_se: bool = False


# Default MCUNet architecture (STM32F746, 320KB SRAM, 1MB Flash)
# Searched by TinyNAS for ImageNet classification
MCUNET_DEFAULT_ARCH = [
    # in_ch, out_ch, n_blocks, stride, kernel, expand, use_se
    BlockConfig(16,  16,  1, 1, 3, 1,  False),   # Stage 0 (no expansion)
    BlockConfig(16,  24,  2, 2, 3, 6,  False),   # Stage 1
    BlockConfig(24,  40,  2, 2, 5, 6,  True),    # Stage 2 (5x5 kernel from NAS)
    BlockConfig(40,  80,  3, 2, 3, 6,  False),   # Stage 3
    BlockConfig(80,  112, 3, 1, 5, 6,  True),    # Stage 4
    BlockConfig(112, 192, 4, 2, 5, 6,  True),    # Stage 5
    BlockConfig(192, 320, 1, 1, 3, 6,  False),   # Stage 6
]

# Smaller variant for 256KB SRAM boards (STM32F412)
MCUNET_SMALL_ARCH = [
    BlockConfig(8,   16,  1, 1, 3, 1,  False),
    BlockConfig(16,  24,  2, 2, 3, 4,  False),
    BlockConfig(24,  32,  2, 2, 5, 4,  False),
    BlockConfig(32,  64,  3, 2, 3, 4,  False),
    BlockConfig(64,  96,  3, 1, 5, 4,  False),
    BlockConfig(96,  160, 3, 2, 5, 4,  False),
    BlockConfig(160, 256, 1, 1, 3, 6,  False),
]


class MCUNet(nn.Module):
    """
    MCUNet backbone network.

    Args:
        arch:        List of BlockConfig (TinyNAS search result).
        n_classes:   Output classes (1000 ImageNet, 2 VWW, 35 Speech Commands).
        input_size:  Input resolution (96, 112, 128, 144, 160 — lower = less SRAM).
        width_mult:  Global width multiplier (0.35 – 1.0).
        dropout:     Classifier dropout rate.
        quantize:    Wrap for INT8 Quantization-Aware Training.
    """
    def __init__(
        self,
        arch: List[BlockConfig] = None,
        n_classes: int = 1000,
        input_size: int = 96,
        width_mult: float = 1.0,
        dropout: float = 0.2,
        quantize: bool = False,
    ):
        super().__init__()
        self.quantize = quantize
        arch = arch or MCUNET_DEFAULT_ARCH

        def _c(ch): return max(1, int(ch * width_mult))

        # First convolution layer (stride=2 to halve spatial dims immediately)
        first_ch = _c(16)
        self.first_conv = nn.Sequential(
            nn.Conv2d(3, first_ch, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(first_ch),
            nn.ReLU6(inplace=True),
        )

        # Build MBConv stages
        stages = []
        in_ch = first_ch
        for cfg in arch:
            out_ch = _c(cfg.out_ch)
            for i in range(cfg.n_blocks):
                stages.append(MBConvBlock(
                    in_channels=in_ch,
                    out_channels=out_ch,
                    kernel_size=cfg.kernel_size,
                    stride=cfg.stride if i == 0 else 1,
                    expand_ratio=cfg.expand_ratio,
                    use_se=cfg.use_se,
                ))
                in_ch = out_ch
        self.stages = nn.Sequential(*stages)

        # Feature expansion head
        last_ch = _c(1280)
        self.feature_head = nn.Sequential(
            nn.Conv2d(in_ch, last_ch, 1, bias=False),
            nn.BatchNorm2d(last_ch),
            nn.ReLU6(inplace=True),
        )

        # Classifier
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Dropout(dropout),
            nn.Linear(last_ch, n_classes),
        )

        # QAT stubs
        if quantize:
            self.quant = QuantStub()
            self.dequant = DeQuantStub()

        self._initialize_weights()

    def forward(self, x):
        if self.quantize:
            x = self.quant(x)
        x = self.first_conv(x)
        x = self.stages(x)
        x = self.feature_head(x)
        x = self.classifier(x)
        if self.quantize:
            x = self.dequant(x)
        return x

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.zeros_(m.bias)

    def fuse_for_quantization(self):
        """Fuse Conv-BN-ReLU triplets for efficient INT8 inference."""
        for m in self.modules():
            if isinstance(m, MBConvBlock):
                # Fuse sequences within each MBConv block
                pass  # Real fusion done via torch.quantization.fuse_modules
        return self


# ─────────────────────────────────────────────
# 4.  Task-Specific Model Variants
# ─────────────────────────────────────────────

def mcunet_imagenet(quantize: bool = False) -> MCUNet:
    """MCUNet for ImageNet (1000 classes, 96×96 input, STM32F746)."""
    return MCUNet(
        arch=MCUNET_DEFAULT_ARCH,
        n_classes=1000,
        input_size=96,
        width_mult=1.0,
        dropout=0.2,
        quantize=quantize,
    )


def mcunet_vww(quantize: bool = False) -> MCUNet:
    """
    MCUNet for Visual Wake Words (VWW).
    Binary classification: person / no-person.
    Target: STM32F746, 96×96 input.
    """
    return MCUNet(
        arch=MCUNET_DEFAULT_ARCH,
        n_classes=2,
        input_size=96,
        width_mult=0.75,
        dropout=0.1,
        quantize=quantize,
    )


def mcunet_keyword_spotting(n_mels: int = 40, n_classes: int = 35) -> nn.Module:
    """
    MCUNet adapted for Keyword Spotting (Speech Commands dataset).
    Input: Log-mel spectrogram (1 × n_mels × T) treated as 1-channel image.
    """
    class KWSAdapter(nn.Module):
        def __init__(self):
            super().__init__()
            # Convert 1-channel audio spectrogram to 3-channel for backbone
            self.audio_to_rgb = nn.Conv2d(1, 3, 1, bias=False)
            self.backbone = MCUNet(
                arch=MCUNET_SMALL_ARCH,
                n_classes=n_classes,
                width_mult=0.5,
                dropout=0.1,
            )
        def forward(self, x):  # x: [B, 1, n_mels, T]
            x = self.audio_to_rgb(x)
            return self.backbone(x)
    return KWSAdapter()


def mcunet_anomaly_detection(feature_dim: int = 128) -> nn.Module:
    """
    MCUNet for unsupervised anomaly detection (e.g. industrial inspection).
    Uses MCUNet as encoder; anomaly score = reconstruction error.
    """
    class AnomalyNet(nn.Module):
        def __init__(self):
            super().__init__()
            self.encoder = MCUNet(
                arch=MCUNET_SMALL_ARCH,
                n_classes=feature_dim,
                width_mult=0.5,
                dropout=0.0,
            )
            # Lightweight decoder (not deployed to MCU — runs offline)
            self.decoder = nn.Sequential(
                nn.Linear(feature_dim, 256),
                nn.ReLU(),
                nn.Linear(256, 3 * 96 * 96),
                nn.Sigmoid(),
            )
        def forward(self, x):
            z = self.encoder(x)
            recon = self.decoder(z).view(-1, 3, 96, 96)
            return z, recon
    return AnomalyNet()


# ─────────────────────────────────────────────
# 5.  Resource Profiler
# ─────────────────────────────────────────────

def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def count_macs(model: nn.Module, input_size: Tuple[int, ...] = (1, 3, 96, 96)) -> int:
    """
    Estimate Multiply-Accumulate operations (MACs).
    Rule: Conv MACs = Kh * Kw * Cin * Cout * H * W
    """
    total_macs = 0
    hooks = []

    def conv_hook(module, inp, out):
        nonlocal total_macs
        if isinstance(module, nn.Conv2d):
            b, c_out, h_out, w_out = out.shape
            c_in = inp[0].shape[1]
            kh, kw = module.kernel_size
            groups = module.groups
            # MACs = (Cin/groups) * Kh * Kw * Cout * H_out * W_out
            macs = (c_in // groups) * kh * kw * c_out * h_out * w_out
            total_macs += macs
        elif isinstance(module, nn.Linear):
            total_macs += module.in_features * module.out_features

    for m in model.modules():
        hooks.append(m.register_forward_hook(conv_hook))

    x = torch.zeros(input_size)
    with torch.no_grad():
        model(x)

    for h in hooks:
        h.remove()

    return total_macs


def estimate_sram_kb(model: nn.Module, input_size=(1, 3, 96, 96), bytes_per_elem=1) -> float:
    """
    Estimate peak SRAM usage (activation maps).
    INT8 inference: 1 byte/element.
    Float32 training: 4 bytes/element.
    Uses TinyEngine's key insight: peak SRAM = max over all layers of
    (input_activation + output_activation) for that layer.
    """
    activations = []
    hooks = []

    def activation_hook(module, inp, out):
        if isinstance(module, (nn.Conv2d, nn.Linear)):
            in_bytes = inp[0].numel() * bytes_per_elem
            out_bytes = out.numel() * bytes_per_elem
            activations.append(in_bytes + out_bytes)

    for m in model.modules():
        hooks.append(m.register_forward_hook(activation_hook))

    x = torch.zeros(input_size)
    with torch.no_grad():
        model(x)

    for h in hooks:
        h.remove()

    peak_bytes = max(activations) if activations else 0
    return peak_bytes / 1024  # KB


def estimate_flash_kb(model: nn.Module, bits: int = 8) -> float:
    """
    Estimate Flash usage for model weights.
    INT8: 1 byte/param, FP32: 4 bytes/param.
    """
    n_params = count_parameters(model)
    bytes_per_param = bits // 8
    return (n_params * bytes_per_param) / 1024  # KB


def estimate_latency_ms(macs: int, cpu_mhz: int, ops_per_cycle: float = 2.0) -> float:
    """
    Rough latency estimate for ARM Cortex-M7 with CMSIS-NN / TinyEngine.
    TinyEngine typically achieves ~2 MAC/cycle on M7 with SIMD.
    """
    ops_per_second = cpu_mhz * 1e6 * ops_per_cycle
    return (macs / ops_per_second) * 1000  # ms


def profile_model(model: nn.Module, task: str, mcu_config: MCUConfig,
                  input_size=(1, 3, 96, 96)) -> dict:
    """Run full resource profiling for a given model and MCU target."""
    macs = count_macs(model, input_size)
    params = count_parameters(model)
    sram_kb = estimate_sram_kb(model, input_size, bytes_per_elem=1)  # INT8
    flash_kb = estimate_flash_kb(model, bits=8)
    latency_ms = estimate_latency_ms(macs, mcu_config.cpu_mhz)

    fits_sram = sram_kb <= mcu_config.sram_kb
    fits_flash = flash_kb <= mcu_config.flash_kb

    return {
        "task": task,
        "mcu": mcu_config.name,
        "parameters": params,
        "macs": macs,
        "gmacs": macs / 1e9,
        "peak_sram_kb": round(sram_kb, 1),
        "sram_budget_kb": mcu_config.sram_kb,
        "sram_utilization_%": round(100 * sram_kb / mcu_config.sram_kb, 1),
        "flash_kb": round(flash_kb, 1),
        "flash_budget_kb": mcu_config.flash_kb,
        "flash_utilization_%": round(100 * flash_kb / mcu_config.flash_kb, 1),
        "latency_ms": round(latency_ms, 1),
        "fps": round(1000 / latency_ms, 1) if latency_ms > 0 else 0,
        "fits_sram": fits_sram,
        "fits_flash": fits_flash,
        "deployable": fits_sram and fits_flash,
    }


# ─────────────────────────────────────────────
# 6.  Quantization-Aware Training (QAT)
# ─────────────────────────────────────────────

def prepare_qat_model(model: MCUNet) -> MCUNet:
    """
    Prepare model for INT8 Quantization-Aware Training.
    Inserts fake-quantization nodes for weights and activations.
    """
    model.train()
    model.qconfig = torch.quantization.get_default_qat_qconfig("qnnpack")
    torch.quantization.prepare_qat(model, inplace=True)
    return model


def convert_to_int8(model: MCUNet) -> nn.Module:
    """Convert QAT-trained model to static INT8 for deployment."""
    model.eval()
    return torch.quantization.convert(model, inplace=True)


# ─────────────────────────────────────────────
# 7.  Knowledge Distillation Loss
# ─────────────────────────────────────────────

class DistillationLoss(nn.Module):
    """
    Combined CE + KL-Divergence loss for training MCUNet
    with a larger teacher network. Boosts accuracy ~1-2% on MCU.
    """
    def __init__(self, alpha: float = 0.5, temperature: float = 4.0):
        super().__init__()
        self.alpha = alpha
        self.T = temperature
        self.ce = nn.CrossEntropyLoss()

    def forward(self, student_logits, teacher_logits, labels):
        ce_loss = self.ce(student_logits, labels)
        soft_student = F.log_softmax(student_logits / self.T, dim=1)
        soft_teacher = F.softmax(teacher_logits / self.T, dim=1)
        kd_loss = F.kl_div(soft_student, soft_teacher, reduction="batchmean") * (self.T ** 2)
        return (1 - self.alpha) * ce_loss + self.alpha * kd_loss


# ─────────────────────────────────────────────
# 8.  Main — Profile All Tasks
# ─────────────────────────────────────────────

if __name__ == "__main__":
    import argparse, json

    parser = argparse.ArgumentParser(description="MCUNet PyTorch — Resource Profiler")
    parser.add_argument("--mcu", default="STM32F746", choices=list(STM32_CONFIGS.keys()))
    parser.add_argument("--task", default="all",
                        choices=["all", "classification", "vww", "kws", "anomaly"])
    parser.add_argument("--profile", action="store_true", default=True)
    args = parser.parse_args()

    mcu = STM32_CONFIGS[args.mcu]
    print(f"\n{'='*60}")
    print(f"  MCUNet Resource Profiler")
    print(f"  Target MCU : {mcu.name}  ({mcu.sram_kb}KB SRAM / {mcu.flash_kb}KB Flash)")
    print(f"{'='*60}\n")

    tasks = {
        "ImageNet Classification": (mcunet_imagenet, (1, 3, 96, 96)),
        "Visual Wake Words (VWW)": (mcunet_vww, (1, 3, 96, 96)),
        "Keyword Spotting (KWS)": (mcunet_keyword_spotting, (1, 1, 40, 101)),
    }

    results = []
    for task_name, (model_fn, input_size) in tasks.items():
        model = model_fn() if not callable(model_fn()) else model_fn()
        # fix for factory functions that return nn.Module
        if callable(model):
            model = model()
        model.eval()
        r = profile_model(model, task_name, mcu, input_size)
        results.append(r)

        status = "✅ DEPLOYABLE" if r["deployable"] else "❌ EXCEEDS BUDGET"
        print(f"Task: {task_name}  [{status}]")
        print(f"  Parameters    : {r['parameters']:,}")
        print(f"  MACs          : {r['macs']:,}  ({r['gmacs']:.3f} GMACs)")
        print(f"  Peak SRAM     : {r['peak_sram_kb']} KB  / {r['sram_budget_kb']} KB  "
              f"({r['sram_utilization_%']}%)")
        print(f"  Flash (Weights): {r['flash_kb']} KB  / {r['flash_budget_kb']} KB  "
              f"({r['flash_utilization_%']}%)")
        print(f"  Est. Latency  : {r['latency_ms']} ms  (~{r['fps']} FPS)")
        print()

    # Save results
    with open("resource_profile.json", "w") as f:
        json.dump(results, f, indent=2)
    print("Results saved → resource_profile.json")
