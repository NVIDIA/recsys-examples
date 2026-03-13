FROM base
ENV PYTHONPATH=:/workspace/recsys-examples/examples/
VOLUME /workspace/recsys-examples
WORKDIR /workspace/recsys-examples/examples/hstu
CMD ["python3", "./inference/inference_gr_ranking.py", "--gin_config_file", "./inference/configs/kuairand_1k_inference_ranking.gin", "--checkpoint_dir", "xxx", "--mode", "snap"]