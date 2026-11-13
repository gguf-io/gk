# Evidence - LLaDA-Image-Turbo 512x512, one denoising step

Machine: macOS 15.7, Apple Clang 17, AMD Radeon RX 580 eGPU (8 GB,
hasUnifiedMemory = NO, simdgroup width 64, Metal 3).

Same binary, only GK_METAL_STORAGE differs:

| prompt | storage | sampling | VAE decode | image bytes | md5 |
|---|---|---|---|---|---|
| a pig in sunglasses | shared | 340.47 s | 145.41 s | 631487 | f058dfde3534de9953a6e5a6b62b4c5a |
| a pig in sunglasses | private | 98.46 s | 15.86 s | 631487 | f058dfde3534de9953a6e5a6b62b4c5a |

Three prompts on the fixed path:

| prompt | sampling | VAE | GPU mem load/peak | image bytes | md5 |
|---|---|---|---|---|---|
| a pig in sunglasses | 98.46 s | 15.86 s | 3.33 / 3.90 GiB | 631487 | f058dfde3534de9953a6e5a6b62b4c5a |
| a red fox standing in fresh snow, soft winter light | 98.90 s | 16.09 s | 3.33 / 3.90 GiB | 645058 | 8a41e753e89a5b94de568f507d24917d |
| a cyberpunk city street at night in the rain, neon signs | 93.16 s | 15.90 s | 3.33 / 3.90 GiB | 752431 | 1e3bdbb0cbc9d3bad2142dc9e46c3d5d |

GPU memory is inUseVidMemoryBytes from ioreg; card total 8.19 GiB.

Command shape (from the engine root, with the eGPU shim loaded):

    DYLD_INSERT_LIBRARIES=<shim>/egpu_shim.dylib \
    diffusion --diffusion-model LLaDA-image-turbo-nvfp4.gguf \
      --vae pig_flux2_vae_fp32-f16.gguf --llm pig_clip-int8.gguf \
      --llm-adapter pig_llada_adapter-f16.gguf \
      -p "<prompt>" --cfg-scale 1.00 --steps 1 -W 512 -H 512 \
      --sampling-method euler --diffusion-fa --backend te=cpu --max-vram 6 \
      -o out.png

Images: pig.private.png, fox.private.png, city.private.png.
