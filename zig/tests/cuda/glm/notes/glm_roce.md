# glm_roce.cu notes

## protocol

One-shot RoCE all-gather between DGX Sparks: the GPU side of b12x's "RoCEnante" protocol
(https://github.com/local-inference-lab/b12x, b12x/comm/roce/_allgather_cute.py at commit
8a99d639410e39d5f39cb4037675331beceea1d4, Copyright the b12x contributors, Apache License 2.0),
reimplemented in CUDA C++ from its CuTe DSL kernel with the same stages and memory orderings. This is TensorFold v1.10's
tensorfold/cuda/roce.cu (gather_kernel) without the ATen launcher: the engine launches it through the driver API.

The pinned host region (roce_proxy.c) holds, per rank: recv[src][slot], flag[src][slot][hca], send[slot] and a
control record {seq, nbytes, error seq, missing peer, nbytes of slot 0, nbytes of slot 1, missing hca}. One launch:

1. every block stages its share of the local shard into send[seq & 1];
2. the last block to finish staging (a free-running arrival counter) publishes nbytes and rings ctrl.seq after
   a system fence; the proxy thread RDMA-writes the payload to every peer, then a flag per HCA stripe;
3. one thread per (peer, HCA) spins on that stripe's flag with system-scope acquire loads (a limit poisons);
4. every block copies the shards in rank order: its own from the input, the peers' from the NIC-written slots;
5. the last block to finish advances the device epoch, so CUDA graphs replay the exchange.
