# qdump

v1.8's TF_GLM_DENSE=q4 load of a few tensors of one rank, dumped as raw bytes for the Zig port's `tf-glm qdump`:
`qdump.py MODEL_DIR RANK OUT_DIR [NAME ...]`. Run on the GPU in the v1.8 image (PYTHONPATH=py18/src): the quantizer's
float sums depend on torch's CUDA reduction order. Each tensor NAME gets NAME.w, NAME.s, NAME.b (Q4: the tiled int32
words, bf16 scales and biases) or NAME.q8, NAME.sc (F8: e4m3 bytes, fp32 scales); a stack is the loader's
(qmm.stack_dense_q4).
