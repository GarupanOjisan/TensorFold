// Adapted from b12x's b12x/comm/roce/_roce_proxy.c at 8a99d639410e (Apache-2.0); notes/roce_proxy.md#origin

// The RDMA proxy thread of the one-shot RoCE exchange; region layout and protocol: notes/roce_proxy.md#layout

#define _GNU_SOURCE
#include <errno.h>
#include <infiniband/verbs.h>
#include <pthread.h>
#include <sched.h>
#include <stdatomic.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#define ROCE_MAX_PEERS 16
#define ROCE_MAX_HCAS 4
#define ROCE_SLOTS 2
#define ROCE_FLAG_STRIDE 128
#define ROCE_PORT 1
#define ROCE_SEND_DEPTH 256
#define ROCE_ABI_VERSION 5
// Model graphs leave sub-millisecond gaps between collectives: stay hot across them (a sleep costs a wakeup)
#define ROCE_IDLE_SPINS 20000000

typedef struct {
    uint64_t region_addr;
    uint32_t rkey[ROCE_MAX_HCAS];
    uint16_t lid[ROCE_MAX_HCAS];
    uint8_t gid[ROCE_MAX_HCAS][16];
    uint32_t mtu[ROCE_MAX_HCAS];
    uint32_t qp_num[ROCE_MAX_HCAS][ROCE_MAX_PEERS];
} roce_blob_t;

typedef struct {
    struct ibv_context *ctx;
    struct ibv_pd *pd;
    struct ibv_mr *mr;
    struct ibv_cq *cq;
    struct ibv_qp *qp[ROCE_MAX_PEERS];
    uint32_t outstanding[ROCE_MAX_PEERS];
    uint64_t writes_completed;
    uint64_t bytes_posted;
    union ibv_gid gid;
    uint16_t lid;
    enum ibv_mtu mtu;
} roce_hca_t;

typedef struct {
    int world;
    int rank;
    int n_hca;
    int gid_index[ROCE_MAX_HCAS];
    int traffic_class;
    // routes: stripe s to peer p leaves on local HCA stripe_hca[p][s] for the peer's HCA stripe_remote[p][s]
    int n_stripe[ROCE_MAX_PEERS];
    int stripe_hca[ROCE_MAX_PEERS][ROCE_MAX_HCAS];
    int stripe_remote[ROCE_MAX_PEERS][ROCE_MAX_HCAS];
    int n_flag;                      // stripes (flags) a peer, the same for every peer and on every rank
    roce_hca_t hca[ROCE_MAX_HCAS];
    uint8_t *region;
    size_t region_bytes;
    size_t slot_bytes;
    size_t recv_off;
    size_t flag_off;
    size_t send_off;
    size_t ctrl_off;
    int started;
    uint64_t peer_addr[ROCE_MAX_PEERS];
    uint32_t peer_rkey[ROCE_MAX_HCAS][ROCE_MAX_PEERS];   // by stripe
    pthread_t thread;
    atomic_int running;
    atomic_int failed;
    uint32_t last_seq;
    uint64_t ops_posted;
    uint64_t writes_completed;
    char err[512];
} roce_ctx_t;

void roce_destroy(roce_ctx_t *c);

static void set_err(roce_ctx_t *c, const char *what, int e) {
    snprintf(c->err, sizeof(c->err), "%s: %s", what, e ? strerror(e) : "failed");
}

int roce_abi_version(void) { return ROCE_ABI_VERSION; }

int roce_layout(int world, uint64_t slot_bytes, uint64_t *out) {
    // out = {recv_off, flag_off, send_off, ctrl_off, total_bytes, flag_stride, slots}
    if (world < 2 || world > ROCE_MAX_PEERS || slot_bytes == 0 || (slot_bytes % 4096) != 0) {
        return -1;
    }
    // Reject a layout whose arithmetic would wrap, so a wrapped region fails here rather than at the NIC
    uint64_t recv_bytes, flag_bytes, send_bytes, send_off, ctrl_off, total;
    if (slot_bytes > ((uint64_t)1 << 40) ||
        __builtin_mul_overflow((uint64_t)world * ROCE_SLOTS, slot_bytes, &recv_bytes) ||
        __builtin_mul_overflow(
            (uint64_t)world * ROCE_SLOTS * ROCE_MAX_HCAS,
            (uint64_t)ROCE_FLAG_STRIDE,
            &flag_bytes) ||
        __builtin_mul_overflow((uint64_t)ROCE_SLOTS, slot_bytes, &send_bytes) ||
        __builtin_add_overflow(recv_bytes, flag_bytes, &send_off) ||
        __builtin_add_overflow(send_off, send_bytes, &ctrl_off) ||
        __builtin_add_overflow(ctrl_off, (uint64_t)ROCE_FLAG_STRIDE, &total)) {
        return -1;
    }
    uint64_t recv_off = 0;
    uint64_t flag_off = recv_off + recv_bytes;
    out[0] = recv_off;
    out[1] = flag_off;
    out[2] = send_off;
    out[3] = ctrl_off;
    out[4] = total;
    out[5] = ROCE_FLAG_STRIDE;
    out[6] = ROCE_SLOTS;
    return 0;
}

uint64_t roce_blob_bytes(void) { return sizeof(roce_blob_t); }

#include "roce_proxy_hca.h"

static int drain_cq(roce_ctx_t *c, int h) {
    struct ibv_wc wc[32];
    int n = ibv_poll_cq(c->hca[h].cq, 32, wc);
    if (n < 0) {
        set_err(c, "ibv_poll_cq", errno);
        return -1;
    }
    for (int i = 0; i < n; i++) {
        if (wc[i].status != IBV_WC_SUCCESS) {
            snprintf(c->err, sizeof(c->err),
                     "RDMA write to rank %u failed: %s (vendor_err 0x%x, seq %u)",
                     (unsigned)wc[i].wr_id, ibv_wc_status_str(wc[i].status),
                     wc[i].vendor_err, c->last_seq);
            return -1;
        }
        c->hca[h].outstanding[wc[i].wr_id] -= 1;
        c->hca[h].writes_completed += 1;
        c->writes_completed += 1;
    }
    return 0;
}

static int post_op(roce_ctx_t *c, uint32_t seq, uint32_t nbytes) {
    if (nbytes == 0 || nbytes % 16 != 0) {
        snprintf(c->err, sizeof(c->err),
                 "RoCE payload size must be a positive multiple of 16 bytes, got %u",
                 nbytes);
        return -1;
    }
    uint32_t slot = seq & 1u;
    uint8_t *send = c->region + c->send_off + (size_t)slot * c->slot_bytes;
    uint32_t seq_copy = seq;
    uint32_t total_packs = nbytes / 16;
    for (int p = 0; p < c->world; p++) {
        if (p == c->rank) {
            continue;
        }
        uint32_t pack_offset = 0;
        const int n_stripe = c->n_stripe[p];
        for (int s = 0; s < n_stripe; s++) {
            const int h = c->stripe_hca[p][s];
            roce_hca_t *hca = &c->hca[h];
            // Two work requests a stripe; the queue at most a quarter full, so a provider's extra entries fit
            while (hca->outstanding[p] >= ROCE_SEND_DEPTH / 4) {
                if (drain_cq(c, h) != 0) {
                    return -1;
                }
                // A peer that stopped acknowledging keeps the QP retrying: honour a stop instead of blocking teardown
                if (!atomic_load_explicit(&c->running, memory_order_relaxed)) {
                    snprintf(c->err, sizeof(c->err),
                             "RoCE proxy stopped with %u writes outstanding to rank %d",
                             hca->outstanding[p], p);
                    return -1;
                }
            }
            uint32_t stripe_packs = total_packs / (uint32_t)n_stripe;
            if ((uint32_t)s < total_packs % (uint32_t)n_stripe) {
                stripe_packs += 1;
            }
            uint32_t stripe_bytes = stripe_packs * 16;
            uint64_t byte_offset = (uint64_t)pack_offset * 16;
            uint64_t remote = c->peer_addr[p];
            struct ibv_sge flag_sge = {
                .addr = (uint64_t)(uintptr_t)&seq_copy,
                .length = 4,
                .lkey = 0,
            };
            struct ibv_send_wr flag_wr;
            memset(&flag_wr, 0, sizeof(flag_wr));
            flag_wr.wr_id = (uint64_t)p;
            flag_wr.sg_list = &flag_sge;
            flag_wr.num_sge = 1;
            flag_wr.opcode = IBV_WR_RDMA_WRITE;
            flag_wr.send_flags = IBV_SEND_SIGNALED | IBV_SEND_INLINE;
            flag_wr.wr.rdma.remote_addr =
                remote + c->flag_off +
                (((uint64_t)c->rank * ROCE_SLOTS + slot) * (uint64_t)c->n_flag +
                 (uint64_t)s) * ROCE_FLAG_STRIDE;
            flag_wr.wr.rdma.rkey = c->peer_rkey[s][p];

            struct ibv_send_wr data_wr;
            struct ibv_sge data_sge;
            struct ibv_send_wr *first_wr = &flag_wr;
            if (stripe_bytes != 0) {
                data_sge = (struct ibv_sge){
                    .addr = (uint64_t)(uintptr_t)(send + byte_offset),
                    .length = stripe_bytes,
                    .lkey = hca->mr->lkey,
                };
                memset(&data_wr, 0, sizeof(data_wr));
                data_wr.wr_id = (uint64_t)p;
                data_wr.next = &flag_wr;
                data_wr.sg_list = &data_sge;
                data_wr.num_sge = 1;
                data_wr.opcode = IBV_WR_RDMA_WRITE;
                data_wr.wr.rdma.remote_addr =
                    remote + c->recv_off +
                    ((uint64_t)c->rank * ROCE_SLOTS + slot) * c->slot_bytes +
                    byte_offset;
                data_wr.wr.rdma.rkey = c->peer_rkey[s][p];
                first_wr = &data_wr;
            }
            struct ibv_send_wr *bad = NULL;
            int rc = ibv_post_send(hca->qp[p], first_wr, &bad);
            if (rc != 0) {
                set_err(c, "ibv_post_send", rc);
                return -1;
            }
            hca->outstanding[p] += 1;
            hca->bytes_posted += stripe_bytes;
            pack_offset += stripe_packs;
        }
    }
    c->ops_posted += 1;
    for (int h = 0; h < c->n_hca; h++) {
        if (drain_cq(c, h) != 0) {
            return -1;
        }
    }
    return 0;
}

static void *proxy_main(void *arg) {
    roce_ctx_t *c = (roce_ctx_t *)arg;
    volatile uint32_t *ctrl = (volatile uint32_t *)(c->region + c->ctrl_off);
    // Spin while ops flow, nap between polls after ROCE_IDLE_SPINS idle ones; notes/roce_proxy.md#idle
    uint64_t idle = 0;
    const struct timespec nap = {0, 20000};
    while (atomic_load_explicit(&c->running, memory_order_relaxed)) {
        uint32_t seq = __atomic_load_n(&ctrl[0], __ATOMIC_ACQUIRE);
        if (seq == c->last_seq) {
            idle++;
            if (idle % 64 == 0) {
                for (int h = 0; h < c->n_hca; h++) {
                    if (drain_cq(c, h) != 0) {
                        atomic_store(&c->failed, 1);
                        return NULL;
                    }
                }
            }
            if (idle >= ROCE_IDLE_SPINS) {
                nanosleep(&nap, NULL);
            }
            continue;
        }
        idle = 0;
        // Post every missed doorbell in order (at most ROCE_SLOTS pending); notes/roce_proxy.md#missed-doorbells
        uint32_t pending = seq - c->last_seq;
        if (pending > ROCE_SLOTS) {
            snprintf(c->err, sizeof(c->err),
                     "doorbell skipped %u ops (last %u, now %u)", pending, c->last_seq, seq);
            atomic_store(&c->failed, 1);
            return NULL;
        }
        for (uint32_t s = c->last_seq + 1; pending > 0; s++, pending--) {
            uint32_t nbytes = ctrl[4 + (s & 1u)];
            if (post_op(c, s, nbytes) != 0) {
                atomic_store(&c->failed, 1);
                return NULL;
            }
            c->last_seq = s;
        }
    }
    return NULL;
}

int roce_start(roce_ctx_t *c) {
    if (atomic_load(&c->running)) {
        return 0;
    }
    if (!c->started) {
        // A restart continues from the last posted sequence, so ops rung while stopped are still posted
        volatile uint32_t *ctrl = (volatile uint32_t *)(c->region + c->ctrl_off);
        c->last_seq = ctrl[0];
        c->started = 1;
    }
    atomic_store(&c->failed, 0);
    atomic_store(&c->running, 1);
    int rc = pthread_create(&c->thread, NULL, proxy_main, c);
    if (rc != 0) {
        atomic_store(&c->running, 0);
        set_err(c, "pthread_create", rc);
        return -1;
    }
    return 0;
}

void roce_stop(roce_ctx_t *c) {
    if (atomic_exchange(&c->running, 0)) {
        pthread_join(c->thread, NULL);
    }
}

int roce_failed(roce_ctx_t *c) { return atomic_load(&c->failed); }

const char *roce_error(roce_ctx_t *c) { return c->err; }

uint64_t roce_stat(roce_ctx_t *c, int which) {
    switch (which) {
    case 0:
        return c->ops_posted;
    case 1:
        return c->writes_completed;
    case 2:
        return c->last_seq;
    default:
        return 0;
    }
}

uint64_t roce_hca_stat(roce_ctx_t *c, int hca, int which) {
    if (hca < 0 || hca >= c->n_hca) {
        return -1;
    }
    switch (which) {
    case 0:
        return c->hca[hca].writes_completed;
    case 1:
        return c->hca[hca].bytes_posted;
    default:
        return 0;
    }
}

void roce_destroy(roce_ctx_t *c) {
    if (c == NULL) {
        return;
    }
    roce_stop(c);
    for (int h = 0; h < ROCE_MAX_HCAS; h++) {
        roce_hca_t *hca = &c->hca[h];
        for (int p = 0; p < ROCE_MAX_PEERS; p++) {
            if (hca->qp[p] != NULL) {
                ibv_destroy_qp(hca->qp[p]);
            }
        }
        if (hca->cq != NULL) {
            ibv_destroy_cq(hca->cq);
        }
        if (hca->mr != NULL) {
            ibv_dereg_mr(hca->mr);
        }
        if (hca->pd != NULL) {
            ibv_dealloc_pd(hca->pd);
        }
        if (hca->ctx != NULL) {
            ibv_close_device(hca->ctx);
        }
    }
    free(c);
}
