"""
MCUNet → STM32 Deployment Pipeline
====================================
Complete pipeline to go from trained PyTorch MCUNet model
to a running application on STM32 boards via TinyEngine/X-CUBE-AI.

Steps covered:
  Step 1: Export PyTorch → ONNX
  Step 2: ONNX → TFLite INT8 (with representative dataset calibration)
  Step 3: TFLite → C arrays (for embedding in STM32CubeIDE project)
  Step 4: TinyEngine C code generation stubs
  Step 5: STM32 HAL + UART inference demo code (C)
  Step 6: Performance benchmark script

Requirements (host PC):
    pip install torch onnx onnxruntime tensorflow onnx-tf
    STM32CubeIDE >= 1.12  (for X-CUBE-AI)
    arm-none-eabi-gcc (for cross-compilation checks)
"""

import os
import sys
import json
import struct
import argparse
import numpy as np
from pathlib import Path
import torch
import torch.nn as nn

from .mcunet_pytorch import (
    mcunet_vww, mcunet_imagenet, mcunet_keyword_spotting,
    STM32_CONFIGS, profile_model
)


# ─────────────────────────────────────────────────────
# STEP 1: Export PyTorch → ONNX
# ─────────────────────────────────────────────────────

def export_to_onnx(model: nn.Module, save_path: str,
                   input_shape=(1, 3, 96, 96)) -> str:
    """
    Export trained PyTorch model to ONNX format.
    ONNX is the interchange format before TFLite conversion.
    """
    model.eval()
    dummy = torch.zeros(input_shape)
    onnx_path = save_path.replace(".pth", ".onnx")

    torch.onnx.export(
        model,
        dummy,
        onnx_path,
        export_params=True,
        opset_version=11,         # Opset 11 has best MCU backend support
        do_constant_folding=True, # Fold constants for smaller model
        input_names=["input"],
        output_names=["output"],
        dynamic_axes={
            "input": {0: "batch_size"},
            "output": {0: "batch_size"},
        },
    )
    print(f"[✓] ONNX model saved → {onnx_path}")
    return onnx_path


# ─────────────────────────────────────────────────────
# STEP 2: INT8 Quantization with Representative Dataset
# ─────────────────────────────────────────────────────

def create_representative_dataset(n_samples: int = 100,
                                   input_shape=(96, 96, 3)):
    """
    Generator function for TFLite INT8 calibration.
    In production: use real images from your dataset.
    Here: synthetic random samples for demo.
    """
    def representative_data_gen():
        for _ in range(n_samples):
            # Normalize to [0, 1] matching training preprocessing
            sample = np.random.rand(1, *input_shape).astype(np.float32)
            yield [sample]
    return representative_data_gen


def convert_onnx_to_tflite_int8(onnx_path: str, output_path: str,
                                  input_shape=(1, 3, 96, 96)) -> str:
    """
    Convert ONNX → TFLite with full INT8 quantization.
    Requires: onnx-tf and tensorflow installed.

    INT8 quantization effects:
        - Weights: FP32 (4 bytes) → INT8 (1 byte) → ~4× Flash reduction
        - Activations: FP32 → INT8 → ~4× SRAM reduction
        - Speed: ~2-3× faster on MCUs with INT8 SIMD (Cortex-M7 DSP)
    """
    try:
        import tensorflow as tf
        from onnx_tf.backend import prepare
        import onnx

        # ONNX → TensorFlow SavedModel
        onnx_model = onnx.load(onnx_path)
        tf_rep = prepare(onnx_model)
        tf_saved_path = output_path.replace(".tflite", "_saved_model")
        tf_rep.export_graph(tf_saved_path)
        print(f"[✓] TF SavedModel → {tf_saved_path}")

        # TensorFlow SavedModel → TFLite INT8
        converter = tf.lite.TFLiteConverter.from_saved_model(tf_saved_path)
        converter.optimizations = [tf.lite.Optimize.DEFAULT]

        h, w = input_shape[2], input_shape[3]
        converter.representative_dataset = create_representative_dataset(
            input_shape=(h, w, 3)
        )
        converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
        converter.inference_input_type  = tf.int8
        converter.inference_output_type = tf.int8

        tflite_model = converter.convert()
        Path(output_path).write_bytes(tflite_model)
        size_kb = len(tflite_model) / 1024
        print(f"[✓] TFLite INT8 model saved → {output_path}  ({size_kb:.1f} KB)")
        return output_path

    except ImportError as e:
        print(f"[!] TFLite conversion requires tensorflow & onnx-tf: {e}")
        print("    Run: pip install tensorflow onnx-tf onnx")
        # Write mock .tflite for demo purposes
        mock_path = output_path.replace(".tflite", "_mock.bin")
        Path(mock_path).write_bytes(b"\x00" * 1024)
        return mock_path


# ─────────────────────────────────────────────────────
# STEP 3: TFLite → C Array (for embedding in firmware)
# ─────────────────────────────────────────────────────

def tflite_to_c_array(tflite_path: str, output_dir: str,
                       var_name: str = "mcunet_model") -> None:
    """
    Convert .tflite binary to a C header + source file pair.
    These files are included directly in the STM32CubeIDE project.
    """
    model_bytes = Path(tflite_path).read_bytes()
    n = len(model_bytes)

    header = f"""/**
 * MCUNet Model — Auto-generated C Array
 * Source: {tflite_path}
 * Size  : {n} bytes ({n/1024:.1f} KB)
 *
 * Usage in STM32 main.c:
 *   #include "{var_name}.h"
 *   // Pass to TinyEngine / X-CUBE-AI interpreter
 */

#ifndef {var_name.upper()}_H
#define {var_name.upper()}_H

#include <stdint.h>

#define {var_name.upper()}_SIZE {n}U
#define {var_name.upper()}_VERSION "MCUNet-v1-INT8"

extern const uint8_t {var_name}[];
extern const unsigned int {var_name}_len;

#endif /* {var_name.upper()}_H */
"""

    # C source: hexdump of model bytes
    hex_rows = []
    for i in range(0, n, 16):
        chunk = model_bytes[i:i+16]
        row = ", ".join(f"0x{b:02X}" for b in chunk)
        hex_rows.append(f"  {row},")

    source = f"""/**
 * MCUNet Model Data
 * Auto-generated — do not edit manually.
 */

#include "{var_name}.h"

/* Store model in Flash (read-only) using __attribute__ */
__attribute__((aligned(4)))
const uint8_t {var_name}[{n}U] = {{
{chr(10).join(hex_rows)}
}};

const unsigned int {var_name}_len = {n}U;
"""

    os.makedirs(output_dir, exist_ok=True)
    (Path(output_dir) / f"{var_name}.h").write_text(header, encoding="utf-8")
    (Path(output_dir) / f"{var_name}.c").write_text(source, encoding="utf-8")
    print(f"[✓] C array files → {output_dir}/{var_name}.h / .c")


# ─────────────────────────────────────────────────────
# STEP 4: Generate STM32 Main Application (C)
# ─────────────────────────────────────────────────────

STM32_MAIN_C = '''/**
 * MCUNet Inference Demo — STM32F746 / STM32H743
 * =================================================
 * Runs INT8 MCUNet model via TinyEngine inference library.
 *
 * Hardware connections:
 *   - OV7670 / Arducam Mini 2MP camera on SPI1
 *   - UART2 (PA2/PA3) at 115200 baud for debug output
 *   - LED (PB0) for inference result indicator
 *   - Optional: ILI9341 LCD on SPI2 for live preview
 *
 * Build: STM32CubeIDE 1.12+, add TinyEngine library to project
 * Flash: st-flash write mcunet_demo.bin 0x08000000
 */

#include "main.h"
#include "mcunet_model.h"         /* Auto-generated C array (Step 3) */
#include "tinyengine/tinyengine.h" /* TinyEngine inference library    */
#include "camera/camera_driver.h"  /* OV7670/Arducam driver           */
#include <stdio.h>
#include <string.h>

/* ── Configuration ─────────────────────────────────── */
#define INPUT_WIDTH   96
#define INPUT_HEIGHT  96
#define INPUT_CHANNEL 3
#define INPUT_SIZE    (INPUT_WIDTH * INPUT_HEIGHT * INPUT_CHANNEL)
#define NUM_CLASSES   2           /* VWW: person / no-person          */

/* Labels for VWW task */
static const char *LABELS[NUM_CLASSES] = {"no_person", "person"};

/* ── Global Buffers (SRAM-mapped) ───────────────────── */
/* Place activation buffer in DTCM RAM for speed (Cortex-M7) */
__attribute__((section(".dtcm")))
static int8_t input_buffer[INPUT_SIZE];

__attribute__((section(".dtcm")))
static int8_t output_buffer[NUM_CLASSES];

static TinyEngine_Handle engine;

/* ── UART Printf Redirect ───────────────────────────── */
extern UART_HandleTypeDef huart2;
int __io_putchar(int ch) {
    HAL_UART_Transmit(&huart2, (uint8_t*)&ch, 1, HAL_MAX_DELAY);
    return ch;
}

/* ── Preprocessing ──────────────────────────────────── */
/**
 * Capture frame from camera, resize to 96x96, convert to INT8.
 * INT8 range: [-128, 127]
 * Zero-point: 0  (symmetric quantization)
 * Scale: 1.0/128  (so float 1.0 → INT8 127)
 */
void capture_and_preprocess(void) {
    uint8_t raw_frame[320 * 240 * 3];  /* Camera native: QVGA RGB888 */
    Camera_CaptureFrame(raw_frame);

    /* Bilinear resize 320x240 → 96x96 (implemented in camera_driver) */
    uint8_t resized[INPUT_SIZE];
    Camera_ResizeBilinear(raw_frame, 320, 240, resized, INPUT_WIDTH, INPUT_HEIGHT);

    /* Normalize uint8 [0,255] → int8 [-128,127] */
    for (int i = 0; i < INPUT_SIZE; i++) {
        input_buffer[i] = (int8_t)(resized[i] - 128);
    }
}

/* ── Post-processing ────────────────────────────────── */
int postprocess_and_get_class(void) {
    int max_idx = 0;
    for (int i = 1; i < NUM_CLASSES; i++) {
        if (output_buffer[i] > output_buffer[max_idx])
            max_idx = i;
    }
    return max_idx;
}

/* ── Main ───────────────────────────────────────────── */
int main(void) {
    /* STM32 HAL init (generated by STM32CubeMX) */
    HAL_Init();
    SystemClock_Config();   /* 216 MHz for STM32F746, 480 MHz for H743 */
    MX_GPIO_Init();
    MX_USART2_UART_Init();
    MX_SPI1_Init();         /* Camera SPI */
    MX_DMA2_Init();         /* DMA for camera capture */

    printf("\\r\\n[MCUNet] Starting...\\r\\n");
    printf("[MCUNet] Model size : %u bytes (%.1f KB)\\r\\n",
           mcunet_model_len, mcunet_model_len / 1024.0f);

    /* Initialize TinyEngine */
    TinyEngine_Status status = TinyEngine_Init(
        &engine,
        mcunet_model,         /* Model C array pointer */
        mcunet_model_len,     /* Model size            */
        input_buffer,         /* Input tensor buffer   */
        output_buffer         /* Output tensor buffer  */
    );

    if (status != TINYENGINE_OK) {
        printf("[ERROR] TinyEngine init failed: %d\\r\\n", status);
        Error_Handler();
    }
    printf("[MCUNet] TinyEngine initialized OK\\r\\n");
    printf("[MCUNet] Input : %dx%dx%d INT8\\r\\n",
           INPUT_WIDTH, INPUT_HEIGHT, INPUT_CHANNEL);
    printf("[MCUNet] Output: %d classes\\r\\n", NUM_CLASSES);

    /* Initialize Camera */
    Camera_Init(INPUT_WIDTH, INPUT_HEIGHT);
    printf("[MCUNet] Camera initialized\\r\\n");

    uint32_t frame_count = 0;

    /* ── Inference Loop ──────────────────────────────── */
    while (1) {
        /* Capture frame and preprocess to INT8 */
        uint32_t t_pre = HAL_GetTick();
        capture_and_preprocess();
        uint32_t t_pre_end = HAL_GetTick();

        /* Run inference */
        uint32_t t_inf = HAL_GetTick();
        TinyEngine_Invoke(&engine);
        uint32_t t_inf_end = HAL_GetTick();

        /* Get result */
        int pred = postprocess_and_get_class();

        /* Log results over UART */
        printf("[Frame %5lu] Prediction: %-12s | "
               "Pre: %2lums | Inf: %3lums | Total: %3lums\\r\\n",
               frame_count,
               LABELS[pred],
               (t_pre_end - t_pre),
               (t_inf_end - t_inf),
               (t_inf_end - t_pre));

        /* LED indicator: ON = person detected */
        HAL_GPIO_WritePin(GPIOB, GPIO_PIN_0,
                          pred == 1 ? GPIO_PIN_SET : GPIO_PIN_RESET);

        frame_count++;
        HAL_Delay(10);  /* ~10ms between frames → up to ~30 FPS theoretically */
    }
}
'''

STM32_LINKER_SCRIPT = """\
/* MCUNet STM32F746 Linker Script — Optimized for TinyEngine */
/* Adds .dtcm section for fast activation buffer (DTCM: 0-cycle access) */

MEMORY
{
    FLASH (rx)     : ORIGIN = 0x08000000, LENGTH = 1024K
    DTCM  (xrw)    : ORIGIN = 0x20000000, LENGTH = 64K
    RAM   (xrw)    : ORIGIN = 0x20010000, LENGTH = 256K
}

SECTIONS
{
    .isr_vector : { KEEP(*(.isr_vector)) } >FLASH
    .text       : { *(.text*) *(.rodata*) } >FLASH
    .data       : { *(.data*) } >RAM AT>FLASH
    .bss        : { *(.bss*) *(COMMON) } >RAM

    /* Activation buffers in fast DTCM RAM */
    .dtcm       : { *(.dtcm) } >DTCM AT>FLASH
}
"""

STM32_CMAKE = """\
# MCUNet STM32 CMakeLists.txt
cmake_minimum_required(VERSION 3.16)
project(mcunet_stm32 C ASM)

set(CMAKE_SYSTEM_NAME Generic)
set(CMAKE_SYSTEM_PROCESSOR arm)
set(CMAKE_C_COMPILER arm-none-eabi-gcc)

set(MCU_FLAGS "-mcpu=cortex-m7 -mfpu=fpv5-d16 -mfloat-abi=hard -mthumb")
set(OPT_FLAGS "-Ofast -ffast-math -funroll-loops")

set(CMAKE_C_FLAGS "${MCU_FLAGS} ${OPT_FLAGS} -DSTM32F746xx")

add_executable(mcunet_demo
    Core/Src/main.c
    Core/Src/mcunet_model.c
    TinyEngine/src/tinyengine.c
    TinyEngine/src/operators/conv2d_int8.c
    TinyEngine/src/operators/depthwise_conv_int8.c
    TinyEngine/src/operators/fully_connected_int8.c
    Drivers/Camera/camera_driver.c
)

target_include_directories(mcunet_demo PRIVATE
    Core/Inc
    TinyEngine/include
    Drivers/Camera
)

# Link script with DTCM section
target_link_options(mcunet_demo PRIVATE
    -T${CMAKE_SOURCE_DIR}/STM32F746_mcunet.ld
    -Wl,-Map=mcunet_demo.map
)
"""


def generate_stm32_project(output_dir: str) -> None:
    """Write all STM32 project files to output directory."""
    d = Path(output_dir)
    (d / "Core/Src").mkdir(parents=True, exist_ok=True)
    (d / "Core/Inc").mkdir(parents=True, exist_ok=True)

    (d / "Core/Src/main.c").write_text(STM32_MAIN_C, encoding="utf-8")
    (d / "STM32F746_mcunet.ld").write_text(STM32_LINKER_SCRIPT, encoding="utf-8")
    (d / "CMakeLists.txt").write_text(STM32_CMAKE, encoding="utf-8")

    readme = """# MCUNet STM32 Project

## Hardware Required
- STM32F746 Discovery Board (or NUCLEO-H743ZI for more memory)
- Arducam Mini 2MP Plus (SPI camera module)
- USB-TTL converter for UART debug output

## Software Setup
```bash
# Install ARM toolchain
sudo apt install gcc-arm-none-eabi binutils-arm-none-eabi

# Clone TinyEngine
git clone https://github.com/mit-han-lab/tinyengine
cp -r tinyengine/TinyEngine ./

# Install Python deps for model conversion
pip install torch onnx tensorflow onnx-tf

# Run conversion pipeline
python stm32_deployment.py
```

## Building in STM32CubeIDE
1. File → Import → Existing Projects
2. Select this directory
3. Right-click project → Properties → C/C++ Build → Settings
4. Add X-CUBE-AI middleware (via STM32CubeMX)
5. Build & Flash (Run → Debug)

## Expected Performance (STM32F746 @ 216 MHz)
| Task         | Latency | FPS  | SRAM   | Flash  |
|--------------|---------|------|--------|--------|
| VWW (96×96)  | ~90ms   | ~11  | 280KB  | 490KB  |
| KWS          | ~45ms   | ~22  | 180KB  | 220KB  |
| ImageNet     | ~200ms  | ~5   | 310KB  | 950KB  |

## Pin Connections (STM32F746 Discovery)
| Signal     | STM32 Pin | Arducam Pin |
|------------|-----------|-------------|
| SPI_SCK    | PA5       | SCK         |
| SPI_MISO   | PA6       | MISO        |
| SPI_MOSI   | PA7       | MOSI        |
| SPI_CS_CAM | PG10      | CS          |
| I2C_SDA    | PB9       | SDA         |
| I2C_SCL    | PB8       | SCL         |
| UART_TX    | PA2       | —           |
| UART_RX    | PA3       | —           |
"""
    (d / "README.md").write_text(readme, encoding="utf-8")
    print(f"[✓] STM32 project structure written → {output_dir}/")


# ─────────────────────────────────────────────────────
# STEP 5: Performance Benchmark Script
# ─────────────────────────────────────────────────────

BENCHMARK_SCRIPT = """\
#!/usr/bin/env python3
"""
BENCHMARK_SCRIPT += '''"""
MCUNet STM32 Benchmark Parser
==============================
Reads UART output from the STM32 board and computes statistics.

Usage:
    python benchmark.py --port /dev/ttyUSB0 --frames 200
"""
import serial, argparse, statistics, time

def benchmark(port: str, baud: int, n_frames: int):
    latencies, fps_vals = [], []
    person_count = 0

    with serial.Serial(port, baud, timeout=2) as ser:
        print(f"Connected to {port} @ {baud} baud")
        ser.write(b"\\r\\n")
        start = time.time()
        collected = 0

        while collected < n_frames:
            line = ser.readline().decode("utf-8", errors="ignore").strip()
            if "Prediction" not in line:
                continue

            parts = line.split("|")
            if len(parts) < 4:
                continue

            pred = parts[1].split(":")[1].strip()
            inf_ms = float(parts[3].split(":")[1].replace("ms", "").strip())
            total_ms = float(parts[4].split(":")[1].replace("ms", "").strip())

            latencies.append(inf_ms)
            fps_vals.append(1000 / total_ms)
            if pred == "person":
                person_count += 1
            collected += 1
            print(f"  [{collected:4d}/{n_frames}] {pred:12s}  inf={inf_ms:.1f}ms")

    elapsed = time.time() - start
    print("\\n=== Benchmark Results ===")
    print(f"  Frames      : {n_frames}")
    print(f"  Elapsed     : {elapsed:.1f}s")
    print(f"  Avg Latency : {statistics.mean(latencies):.1f} ms")
    print(f"  Min Latency : {min(latencies):.1f} ms")
    print(f"  Max Latency : {max(latencies):.1f} ms")
    print(f"  Std Dev     : {statistics.stdev(latencies):.2f} ms")
    print(f"  Avg FPS     : {statistics.mean(fps_vals):.1f}")
    print(f"  Person Det% : {100*person_count/n_frames:.1f}%")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--port",   default="/dev/ttyUSB0")
    ap.add_argument("--baud",   type=int, default=115200)
    ap.add_argument("--frames", type=int, default=100)
    args = ap.parse_args()
    benchmark(args.port, args.baud, args.frames)
'''


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="MCUNet to STM32 deployment helper"
    )
    parser.add_argument(
        "--output_dir",
        default="stm32_project",
        help="Directory where STM32 project artifacts are generated",
    )
    args = parser.parse_args()

    OUT = Path(args.output_dir)
    print("MCUNet → STM32 Deployment Pipeline")
    print("="*50)

    # 1. Load a pretrained/trained model
    print("\n[Step 1] Building MCUNet VWW model...")
    model = mcunet_vww(quantize=False)
    model.eval()

    # Count params
    params = sum(p.numel() for p in model.parameters())
    print(f"  Parameters: {params:,}")

    # 2. Profile
    print("\n[Step 2] Resource profiling...")
    mcu = STM32_CONFIGS["STM32F746"]
    profile = profile_model(model, "VWW", mcu, (1, 3, 96, 96))
    print(f"  Peak SRAM  : {profile['peak_sram_kb']} KB")
    print(f"  Flash      : {profile['flash_kb']} KB")
    print(f"  Latency    : {profile['latency_ms']} ms")
    print(f"  Deployable : {profile['deployable']}")

    # 3. Save dummy weights (in real project: load checkpoint)
    dummy_path = OUT / "mcunet_vww.pth"
    OUT.mkdir(exist_ok=True)
    torch.save(model.state_dict(), dummy_path)
    print(f"\n[Step 3] Weights saved → {dummy_path}")

    # 4. Generate STM32 project
    print("\n[Step 4] Generating STM32 project...")
    generate_stm32_project(str(OUT))

    # 5. Write benchmark script
    (OUT / "benchmark.py").write_text(BENCHMARK_SCRIPT, encoding="utf-8")
    print(f"[✓] Benchmark script → {OUT}/benchmark.py")

    # 6. Summary
    print("\n" + "="*50)
    print("DEPLOYMENT CHECKLIST:")
    print("  [1] Train MCUNet: python mcunet_pytorch.py (replace with real data)")
    print("  [2] Export ONNX : export_to_onnx(model, 'mcunet_vww.onnx')")
    print("  [3] INT8 Convert: convert_onnx_to_tflite_int8(...)")
    print("  [4] C Array     : tflite_to_c_array('model.tflite', 'stm32_project/Core/Src/')")
    print("  [5] Build       : Open STM32CubeIDE → Import → Build → Flash")
    print("  [6] Benchmark   : python benchmark.py --port /dev/ttyACM0")
    print("="*50)
