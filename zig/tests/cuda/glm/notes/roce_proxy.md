# roce_proxy.c notes

`zig/src/families/glm/cuda/roce_proxy.c`, with its HCA setup and connection (`open_hca` .. `roce_connect`) in
`roce_proxy_hca.h`, included where that code stood.

## origin

Adapted from b12x (https://github.com/local-inference-lab/b12x), b12x/comm/roce/_roce_proxy.c at commit
8a99d639410e39d5f39cb4037675331beceea1d4: "RoCEnante", Copyright the b12x contributors, Apache License 2.0
(http://www.apache.org/licenses/LICENSE-2.0). Used by tensorfold/cuda/roce.py. Changes from b12x's file (TensorFold,
2026-10-01, ABI 5): up to 4 HCAs, a GID index per HCA, and routes: each peer's stripes go over chosen (local HCA,
peer HCA) pairs (roce_set_route; default HCA h to the peer's HCA h, every HCA), so ranks cabled to each other
directly (a triangle of Sparks: one /24 and two rails a pair) reach each peer over its own rails only. Flags are
indexed by stripe (``n_flag`` stripes a peer, the same for every peer).

## layout

RDMA proxy for the b12x RoCE one-shot all-reduce.

One rank owns one pinned host region laid out as:

```
  recv[src][slot]  (world * SLOTS * slot_bytes)  filled by peers' RDMA writes
  flag[src][slot][hca] sequence number written on each HCA after that HCA's
                        payload stripe
  send[slot]       (SLOTS * slot_bytes)          staged by the local GPU kernel
  ctrl             (FLAG_STRIDE)                 {u32 seq, u32 nbytes, u32 error,
                                                  u32 missing_peer} doorbell; the
                                                 last two are set by the kernel
                                                 when a wait times out
```

The GPU kernel stages its input into send[seq & 1], publishes nbytes and seq in ctrl, then spins on
flag[peer][seq & 1][hca] for every peer and HCA. The proxy thread below spins on ctrl.seq and, for every peer,
stripes the payload across every HCA. Each stripe is followed by its own 4-byte seq write on the same reliable
QP, so its flag cannot become visible before its payload. The GPU waits for every stripe flag before consuming
the receive slot. Nothing on the receive path involves the host.

This file is compiled by b12x.comm.roce._proxy at first use with the host gcc and libibverbs; it must stay plain C
with no CUDA dependency.

`ROCE_IDLE_SPINS`: model graphs leave sub-millisecond gaps between collectives. Keep the proxy hot across those
gaps; sleeping there adds one scheduler wakeup to every collective on the graph's critical path.

## idle

Spin while ops are flowing. After ROCE_IDLE_SPINS polls without a doorbell, request a short nanosleep between
polls (the OS decides the actual delay) so an idle runtime does not hold a core next to the serving process. The
missed-doorbell catch-up below keeps the protocol correct however long the thread is away.

## missed-doorbells

The doorbell holds only the newest sequence. Our kernel for op N completes on the peers' payloads alone, so op
N+1 can ring before this thread has seen op N (it slept, or the scheduler moved it). Peers cannot get further
than one op ahead of us, so at most ROCE_SLOTS doorbells are pending and every send slot is intact: post each
missed sequence in order using its per-slot byte count.
