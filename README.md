# MCUNet on STM32: Analysis, Optimization, and Deployment

This repository provides an end-to-end workflow for:
- Profiling MCUNet resource usage on MCU targets
- Training task-specific models in PyTorch
- Testing architecture improvements for MCU constraints
- Generating STM32 deployment scaffolding

## Professional Project Layout

```text
project/
  src/
    mcunet_project/
      mcunet_pytorch.py      # MCUNet model + profiler
      train.py               # Training pipeline
      improvements.py        # Optimization techniques
      stm32_deployment.py    # Deployment pipeline
      cli.py                 # Unified CLI entrypoint
  scripts/
    bootstrap.ps1            # Create venv + install deps
    quickstart.ps1           # Run profile + improvements + deployment
  configs/                   # Place task or board configs here
  data/                      # Dataset root (ImageFolder expected)
  checkpoints/               # Saved training checkpoints
  artifacts/                 # Exported/generated artifacts
  train.py                   # Backward-compatible launcher
  mcunet_pytorch.py          # Backward-compatible launcher
  improvements.py            # Backward-compatible launcher
  stm32_deployment.py        # Backward-compatible launcher
  requirements.txt
  pyproject.toml
```

## Keil / STM32 Workflow

The code that runs on the board is the C firmware entry point, not the Python scripts. For Keil, use:

- [stm32_project/Core/Src/main.c](stm32_project/Core/Src/main.c)

That file is a Keil-friendly STM32 HAL main loop that initializes UART, optionally a camera, loads the MCUNet model, and runs INT8 inference through TinyEngine.

What to add to your Keil project:

- The single firmware file [stm32_project/Core/Src/main.c](stm32_project/Core/Src/main.c)
- TinyEngine source and include paths only if you replace the embedded placeholder model with a real export
- Your STM32CubeMX-generated `main.h`, `stm32f7xx_hal_*` sources, and board startup files

If you do not have the camera path ready, the code still runs with synthetic input so you can verify the MCU loop, UART logging, and LED control first.

## Full Keil Setup

### 1. Install the tools

- Keil MDK-ARM (uVision)
- STM32CubeMX or STM32CubeIDE for generating the STM32 HAL startup code
- STM32 device pack for your board family, for example STM32F7xx or STM32H7xx
- ST-Link USB driver and ST-LINK Utility or CubeProgrammer for flashing

### 1.5 Check your STM32 family

The current firmware template in this repo is written for STM32F7/H7-style HAL projects and uses a Cortex-M7 setup.

If your board is actually an STM32F1 device, you must replace these board-specific parts before building in Keil:

- `SystemClock_Config()` for the F1 clock tree
- `stm32f7xx_*` headers and sources with `stm32f1xx_*`
- any `.dtcm` memory section usage
- FPU / hard-float settings, because STM32F1 has no FPU
- the startup file and linker script for your exact chip, such as STM32F103x or STM32F12x depending on the real part number

If you are not sure about the exact chip marking, check the full part number printed on the MCU package or board silkscreen.

### 2. Prepare the Keil project

Create a new STM32 project in Keil or import a CubeMX-generated project. The project must already contain:

- `main.h`
- `stm32xxxx_hal_conf.h`
- startup assembly file
- linker script for your STM32 board
- generated HAL sources such as `stm32xxxx_hal_msp.c`

Then replace the generated `main.c` with [stm32_project/Core/Src/main.c](stm32_project/Core/Src/main.c) or copy its contents into your Keil project.

### 3. Add the MCUNet files

The Keil build is now self-contained in one custom file:

- [stm32_project/Core/Src/main.c](stm32_project/Core/Src/main.c)

That file contains the embedded placeholder model bytes and a synthetic-input demo path. If you want real inference, replace the placeholder bytes inside that file with a real exported model and add TinyEngine runtime sources back into the Keil project.

### 4. Configure Keil settings

- Set the correct MCU device in `Options for Target`
- Enable the FPU and floating-point ABI if your STM32 has one
- Add include paths for `Core/Inc`, `TinyEngine/include`, and any camera driver headers
- Add source paths for `Core/Src`, `TinyEngine/src`, and camera driver sources if used
- Make sure the linker script matches the Flash and RAM layout of your board

For the STM32F746 target used in this project, the build should use a Cortex-M7 configuration with hard-float enabled.

### 5. Build and flash

1. Build the project in Keil
2. Connect the board over ST-Link
3. Flash the generated binary or debug directly from Keil
4. Open the serial terminal at 115200 baud to read the inference log

### 5.1 Required files to keep

For a Keil build, the minimum file set is:

- `README.md`
- `pyproject.toml` and `requirements.txt` only if you still want the Python analysis scripts
- `src/mcunet_project/mcunet_pytorch.py`
- `src/mcunet_project/improvements.py`
- `src/mcunet_project/stm32_deployment.py`
- `stm32_project/Core/Src/main.c`
- `mcunet_model.c` and `mcunet_model.h`
- TinyEngine source and header files
- STM32 HAL startup files, linker script, and board support files from CubeMX/Keil

### 6. Run mode

- With camera support enabled, the board captures frames, preprocesses them, and runs MCUNet inference on-device
- Without camera support, the project runs in synthetic-input mode, which is useful for checking that Keil, UART, and the MCU loop are working

## Resource Analysis

The repository already includes MCU-oriented profiling helpers in [src/mcunet_project/mcunet_pytorch.py](src/mcunet_project/mcunet_pytorch.py). The profiler estimates:

- Parameter count
- Peak SRAM from activation maps
- Flash usage from weights
- Rough latency on the target STM32 clock

For the STM32 targets defined in the repo, the default budgets are:

- STM32F746: 320 KB SRAM, 1024 KB Flash, 216 MHz
- STM32H743: 512 KB SRAM, 2048 KB Flash, 480 MHz

The improvement set in [src/mcunet_project/improvements.py](src/mcunet_project/improvements.py) is the right direction for constrained boards:

- Patch-based inference to reduce peak SRAM in early layers
- Binary depthwise weights to reduce Flash
- Taylor-pruned channels to cut MACs and latency
- Dynamic resolution switching to lower average runtime cost

## Notes

- The repo is trimmed to the STM32/Keil path plus the Python analysis code used to size and optimize MCUNet.
- The board-side entry point is [stm32_project/Core/Src/main.c](stm32_project/Core/Src/main.c).
- Generated or PC-only folders were removed to keep the tree focused on the firmware build.
