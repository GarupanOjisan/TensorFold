// roce_proxy.c's HCA setup and connection (open_hca .. roce_connect), included by roce_proxy.c.
static int open_hca(roce_ctx_t *c, int h, const char *name) {
    int num = 0;
    struct ibv_device **list = ibv_get_device_list(&num);
    if (list == NULL) {
        set_err(c, "ibv_get_device_list", errno);
        return -1;
    }
    struct ibv_device *dev = NULL;
    for (int i = 0; i < num; i++) {
        if (strcmp(ibv_get_device_name(list[i]), name) == 0) {
            dev = list[i];
            break;
        }
    }
    if (dev == NULL) {
        ibv_free_device_list(list);
        snprintf(c->err, sizeof(c->err), "RDMA device %s not found", name);
        return -1;
    }
    roce_hca_t *hca = &c->hca[h];
    hca->ctx = ibv_open_device(dev);
    ibv_free_device_list(list);
    if (hca->ctx == NULL) {
        set_err(c, "ibv_open_device", errno);
        return -1;
    }
    struct ibv_port_attr port;
    if (ibv_query_port(hca->ctx, ROCE_PORT, &port) != 0) {
        set_err(c, "ibv_query_port", errno);
        return -1;
    }
    if (port.state != IBV_PORT_ACTIVE) {
        snprintf(c->err, sizeof(c->err), "RDMA device %s port %d is not active", name, ROCE_PORT);
        return -1;
    }
    hca->lid = port.lid;
    hca->mtu = port.active_mtu;
    if (ibv_query_gid(hca->ctx, ROCE_PORT, c->gid_index[h], &hca->gid) != 0) {
        set_err(c, "ibv_query_gid", errno);
        return -1;
    }
    hca->pd = ibv_alloc_pd(hca->ctx);
    if (hca->pd == NULL) {
        set_err(c, "ibv_alloc_pd", errno);
        return -1;
    }
    hca->mr = ibv_reg_mr(hca->pd, c->region, c->region_bytes,
                         IBV_ACCESS_LOCAL_WRITE | IBV_ACCESS_REMOTE_WRITE);
    if (hca->mr == NULL) {
        set_err(c, "ibv_reg_mr(pinned region)", errno);
        return -1;
    }
    hca->cq = ibv_create_cq(hca->ctx, ROCE_SEND_DEPTH * ROCE_MAX_PEERS, NULL, NULL, 0);
    if (hca->cq == NULL) {
        set_err(c, "ibv_create_cq", errno);
        return -1;
    }
    for (int p = 0; p < c->world; p++) {
        if (p == c->rank) {
            continue;
        }
        struct ibv_qp_init_attr attr;
        memset(&attr, 0, sizeof(attr));
        attr.send_cq = hca->cq;
        attr.recv_cq = hca->cq;
        attr.qp_type = IBV_QPT_RC;
        attr.cap.max_send_wr = ROCE_SEND_DEPTH;
        attr.cap.max_recv_wr = 1;
        attr.cap.max_send_sge = 1;
        attr.cap.max_recv_sge = 1;
        attr.cap.max_inline_data = 16;
        hca->qp[p] = ibv_create_qp(hca->pd, &attr);
        if (hca->qp[p] == NULL) {
            set_err(c, "ibv_create_qp", errno);
            return -1;
        }
        struct ibv_qp_attr init;
        memset(&init, 0, sizeof(init));
        init.qp_state = IBV_QPS_INIT;
        init.pkey_index = 0;
        init.port_num = ROCE_PORT;
        init.qp_access_flags = IBV_ACCESS_REMOTE_WRITE;
        int rc = ibv_modify_qp(hca->qp[p], &init,
                               IBV_QP_STATE | IBV_QP_PKEY_INDEX | IBV_QP_PORT | IBV_QP_ACCESS_FLAGS);
        if (rc != 0) {
            set_err(c, "ibv_modify_qp(INIT)", rc);
            return -1;
        }
    }
    return 0;
}

roce_ctx_t *roce_create(int world, int rank, const char *const *hca_names, int n_hca,
                        const int *gid_index, int traffic_class, void *region, uint64_t region_bytes,
                        uint64_t slot_bytes, char *err, uint64_t err_len) {
    uint64_t layout[7];
    if (roce_layout(world, slot_bytes, layout) != 0 || layout[4] > region_bytes ||
        rank < 0 || rank >= world || n_hca < 1 || n_hca > ROCE_MAX_HCAS ||
        traffic_class < 0 || traffic_class > 255) {
        snprintf(err, err_len, "invalid roce runtime geometry");
        return NULL;
    }
    roce_ctx_t *c = calloc(1, sizeof(*c));
    if (c == NULL) {
        snprintf(err, err_len, "out of memory");
        return NULL;
    }
    c->world = world;
    c->rank = rank;
    c->n_hca = n_hca;
    for (int h = 0; h < n_hca; h++) {
        c->gid_index[h] = gid_index[h];
    }
    for (int p = 0; p < world; p++) {           // default routes: HCA h to the peer's HCA h, every HCA
        c->n_stripe[p] = p == rank ? 0 : n_hca;
        for (int h = 0; h < n_hca; h++) {
            c->stripe_hca[p][h] = h;
            c->stripe_remote[p][h] = h;
        }
    }
    c->n_flag = n_hca;
    c->traffic_class = traffic_class;
    c->region = region;
    c->region_bytes = region_bytes;
    c->slot_bytes = slot_bytes;
    c->recv_off = layout[0];
    c->flag_off = layout[1];
    c->send_off = layout[2];
    c->ctrl_off = layout[3];
    for (int h = 0; h < n_hca; h++) {
        if (open_hca(c, h, hca_names[h]) != 0) {
            snprintf(err, err_len, "%s", c->err);
            roce_destroy(c);
            return NULL;
        }
    }
    return c;
}

int roce_set_routes(roce_ctx_t *c, int peer, int n, const int *local, const int *remote, int n_flag) {
    // before roce_connect: peer's stripes (n of them) over (local[s], remote[s]); n_flag: the stripes every peer has
    if (c->started || peer < 0 || peer >= c->world || peer == c->rank || n < 1 || n > ROCE_MAX_HCAS ||
        n != n_flag || n_flag > ROCE_MAX_HCAS) {
        snprintf(c->err, sizeof(c->err), "invalid RoCE route to rank %d (%d stripes, %d a peer)", peer, n, n_flag);
        return -1;
    }
    for (int s = 0; s < n; s++) {
        if (local[s] < 0 || local[s] >= c->n_hca || remote[s] < 0 || remote[s] >= ROCE_MAX_HCAS) {
            snprintf(c->err, sizeof(c->err), "invalid RoCE route to rank %d: stripe %d", peer, s);
            return -1;
        }
        c->stripe_hca[peer][s] = local[s];
        c->stripe_remote[peer][s] = remote[s];
    }
    c->n_stripe[peer] = n;
    c->n_flag = n_flag;
    return 0;
}

int roce_local_blob(roce_ctx_t *c, void *out, uint64_t out_len) {
    if (out_len < sizeof(roce_blob_t)) {
        return -1;
    }
    roce_blob_t blob;
    memset(&blob, 0, sizeof(blob));
    blob.region_addr = (uint64_t)(uintptr_t)c->region;
    for (int h = 0; h < c->n_hca; h++) {
        blob.rkey[h] = c->hca[h].mr->rkey;
        blob.lid[h] = c->hca[h].lid;
        blob.mtu[h] = (uint32_t)c->hca[h].mtu;
        memcpy(blob.gid[h], c->hca[h].gid.raw, 16);
        for (int p = 0; p < c->world; p++) {
            blob.qp_num[h][p] = (p == c->rank) ? 0 : c->hca[h].qp[p]->qp_num;
        }
    }
    memcpy(out, &blob, sizeof(blob));
    return 0;
}

static int connect_qp(roce_ctx_t *c, int h, int p, const roce_blob_t *peer, int rh) {
    // local HCA h's QP for peer p <-> the peer's HCA rh's QP for this rank
    roce_hca_t *hca = &c->hca[h];
    struct ibv_qp_attr rtr;
    memset(&rtr, 0, sizeof(rtr));
    rtr.qp_state = IBV_QPS_RTR;
    rtr.path_mtu = (enum ibv_mtu)(peer->mtu[rh] < (uint32_t)hca->mtu ? peer->mtu[rh] : (uint32_t)hca->mtu);
    rtr.dest_qp_num = peer->qp_num[rh][c->rank];
    rtr.rq_psn = 0;
    rtr.max_dest_rd_atomic = 1;
    rtr.min_rnr_timer = 12;
    rtr.ah_attr.is_global = 1;
    rtr.ah_attr.dlid = peer->lid[rh];
    rtr.ah_attr.sl = 0;
    rtr.ah_attr.src_path_bits = 0;
    rtr.ah_attr.port_num = ROCE_PORT;
    memcpy(rtr.ah_attr.grh.dgid.raw, peer->gid[rh], 16);
    rtr.ah_attr.grh.sgid_index = (uint8_t)c->gid_index[h];
    rtr.ah_attr.grh.hop_limit = 64;
    rtr.ah_attr.grh.traffic_class = (uint8_t)c->traffic_class;
    rtr.ah_attr.grh.flow_label = 0;
    int rc = ibv_modify_qp(hca->qp[p], &rtr,
                           IBV_QP_STATE | IBV_QP_AV | IBV_QP_PATH_MTU | IBV_QP_DEST_QPN |
                               IBV_QP_RQ_PSN | IBV_QP_MAX_DEST_RD_ATOMIC | IBV_QP_MIN_RNR_TIMER);
    if (rc != 0) {
        set_err(c, "ibv_modify_qp(RTR)", rc);
        return -1;
    }
    struct ibv_qp_attr rts;
    memset(&rts, 0, sizeof(rts));
    rts.qp_state = IBV_QPS_RTS;
    rts.timeout = 14;
    rts.retry_cnt = 7;
    rts.rnr_retry = 7;
    rts.sq_psn = 0;
    rts.max_rd_atomic = 1;
    rc = ibv_modify_qp(hca->qp[p], &rts,
                       IBV_QP_STATE | IBV_QP_TIMEOUT | IBV_QP_RETRY_CNT | IBV_QP_RNR_RETRY |
                           IBV_QP_SQ_PSN | IBV_QP_MAX_QP_RD_ATOMIC);
    if (rc != 0) {
        set_err(c, "ibv_modify_qp(RTS)", rc);
        return -1;
    }
    return 0;
}

int roce_connect(roce_ctx_t *c, const void *blobs, uint64_t blobs_len) {
    if (blobs_len < sizeof(roce_blob_t) * (uint64_t)c->world) {
        snprintf(c->err, sizeof(c->err), "peer blob buffer too small");
        return -1;
    }
    const roce_blob_t *all = (const roce_blob_t *)blobs;
    for (int p = 0; p < c->world; p++) {
        if (p == c->rank) {
            continue;
        }
        c->peer_addr[p] = all[p].region_addr;
        if (c->n_stripe[p] != c->n_flag) {
            snprintf(c->err, sizeof(c->err), "rank %d has %d stripes, every peer needs %d", p, c->n_stripe[p],
                     c->n_flag);
            return -1;
        }
        for (int s = 0; s < c->n_stripe[p]; s++) {
            const int h = c->stripe_hca[p][s], rh = c->stripe_remote[p][s];
            c->peer_rkey[s][p] = all[p].rkey[rh];
            if (connect_qp(c, h, p, &all[p], rh) != 0) {
                return -1;
            }
        }
    }
    return 0;
}
