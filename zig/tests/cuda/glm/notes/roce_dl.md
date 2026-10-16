# roce_dl.c notes

libibverbs by dlopen for roce_proxy.c: the 17 entry points it calls, each defined here and resolved from
libibverbs.so.1 at its first call, so a binary that never starts the RoCE transport needs no rdma-core on the
machine. `roce_dl_open` says whether the library loads (call it before roce_create; without it a missing library
aborts at the first verbs call).
