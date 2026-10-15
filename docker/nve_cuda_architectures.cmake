# NVE's native CUDA libraries use NVE_SM_VERSION independently of
# TORCH_CUDA_ARCH_LIST. Preserve the targets used by both supported NVE
# releases, including SM80 for A100, when building without a visible GPU.
set(NVE_SM_VERSION "80;86;89;90;100;120" CACHE STRING "NVE CUDA architectures")
