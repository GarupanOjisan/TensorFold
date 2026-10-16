// libibverbs by dlopen: roce_proxy.c's verbs resolved at first call; call roce_dl_open first; notes/roce_dl.md
#define _GNU_SOURCE
#include <dlfcn.h>
#include <infiniband/verbs.h>
#include <stdio.h>
#include <stdlib.h>

static void *lib;

int roce_dl_open(void) {
    if (lib == NULL) lib = dlopen("libibverbs.so.1", RTLD_NOW | RTLD_LOCAL);
    return lib != NULL;
}

static void *sym(const char *name) {
    void *p = roce_dl_open() ? dlsym(lib, name) : NULL;
    if (p == NULL) {
        fprintf(stderr, "libibverbs: %s unavailable\n", name);
        abort();
    }
    return p;
}

#undef ibv_reg_mr
#undef ibv_query_port
#undef ibv_create_qp
#undef ibv_create_cq
#undef ibv_modify_qp
#undef ibv_query_gid

#define DEF(ret, name, params, args)         \
    ret name params {                        \
        static ret(*f) params;               \
        if (f == NULL) f = sym(#name);       \
        return f args;                       \
    }
#define DEFV(name, params, args)             \
    void name params {                       \
        static void (*f) params;             \
        if (f == NULL) f = sym(#name);       \
        f args;                              \
    }

DEF(struct ibv_device **, ibv_get_device_list, (int *num), (num))
DEFV(ibv_free_device_list, (struct ibv_device **list), (list))
DEF(const char *, ibv_get_device_name, (struct ibv_device *dev), (dev))
DEF(struct ibv_context *, ibv_open_device, (struct ibv_device *dev), (dev))
DEF(int, ibv_close_device, (struct ibv_context *ctx), (ctx))
DEF(int, ibv_query_port, (struct ibv_context *ctx, uint8_t port, struct _compat_ibv_port_attr *attr), (ctx, port, attr))
DEF(int, ibv_query_gid, (struct ibv_context *ctx, uint8_t port, int index, union ibv_gid *gid), (ctx, port, index, gid))
DEF(struct ibv_pd *, ibv_alloc_pd, (struct ibv_context *ctx), (ctx))
DEF(int, ibv_dealloc_pd, (struct ibv_pd *pd), (pd))
DEF(struct ibv_mr *, ibv_reg_mr, (struct ibv_pd *pd, void *addr, size_t length, int access), (pd, addr, length, access))
DEF(int, ibv_dereg_mr, (struct ibv_mr *mr), (mr))
DEF(struct ibv_cq *, ibv_create_cq, (struct ibv_context *ctx, int cqe, void *user, struct ibv_comp_channel *channel, int vector),
    (ctx, cqe, user, channel, vector))
DEF(int, ibv_destroy_cq, (struct ibv_cq *cq), (cq))
DEF(struct ibv_qp *, ibv_create_qp, (struct ibv_pd *pd, struct ibv_qp_init_attr *attr), (pd, attr))
DEF(int, ibv_modify_qp, (struct ibv_qp *qp, struct ibv_qp_attr *attr, int mask), (qp, attr, mask))
DEF(int, ibv_destroy_qp, (struct ibv_qp *qp), (qp))
DEF(const char *, ibv_wc_status_str, (enum ibv_wc_status status), (status))
