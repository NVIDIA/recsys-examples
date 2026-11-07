import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))

# tests/gpu holds the response_logits GPU tests (CUDA + Megatron); tests/integration the slime+Ray test of rshard.
# Both are run by hand, not collected here.
collect_ignore = ["gpu", "integration"]   # integration: needs slime + Ray (run by hand)
