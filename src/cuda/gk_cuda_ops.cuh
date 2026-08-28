#pragma once

// What the backend file needs from the kernel files, and the one conversion
// every launcher starts with.

#include "gk_cuda_common.cuh"

extern "C" {
#include "gk_impl.h"
}

static __host__ __forceinline__ gk_tview gk_cu_view(const struct gk_tensor * t) {
    gk_tview v;
    v.data = (const char *) t->data;
    for (int i = 0; i < 4; ++i) {
        v.ne[i] = t->ne[i];
        v.nb[i] = (int64_t) t->nb[i];
    }
    v.type = (int) t->type;
    return v;
}

static __host__ __forceinline__ gk_tview_mut gk_cu_view_mut(struct gk_tensor * t) {
    gk_tview_mut v;
    v.data = (char *) t->data;
    for (int i = 0; i < 4; ++i) {
        v.ne[i] = t->ne[i];
        v.nb[i] = (int64_t) t->nb[i];
    }
    v.type = (int) t->type;
    return v;
}

static __host__ __forceinline__ int64_t gk_cu_nelements(const struct gk_tensor * t) {
    return t->ne[0] * t->ne[1] * t->ne[2] * t->ne[3];
}

// Evaluates one node. Returns false for an op this backend has no kernel for,
// which never happens for a graph the scheduler placed here - supports_op is
// asked first - and is a loud failure rather than a wrong answer if it does.
//
// `scratch` is the backend's, and belongs to the same stream. It is passed
// rather than reached for globally so that two backends on two devices cannot
// end up sharing one buffer.
bool gk_cuda_compute_op(gkStream_t stream, struct gk_cuda_scratch * scratch,
                        struct gk_tensor * node);

// Whether this backend can evaluate the node, operand types included. The
// scheduler calls this before placing anything.
bool gk_cuda_supports_op(const struct gk_tensor * op);

// The head widths the attention kernel can hold in shared memory. supports_op
// checks against these, so they live here rather than beside the kernel.
#define GK_CUDA_FA_MAX_DK 640
#define GK_CUDA_FA_MAX_DV 512

// The widest head the linear-attention recurrences take. They give each slot
// of a head's state its own thread, so this is a block's thread limit.
#define GK_CUDA_RECURRENT_MAX_S 1024

// The matmuls live in their own translation unit: they are the only kernels
// here with a performance story worth separating out.
void gk_cuda_mul_mat   (gkStream_t stream, struct gk_cuda_scratch * scratch,
                        struct gk_tensor * dst);

// Which of gk_cuda_mul_mat's kernels the last call picked. A shape and a rate
// say a matmul was slow; they do not say whether the fast path declined it, and
// every one of those paths can decline silently - a type it does not cover, a
// scratch allocation that failed, a tile too wide for the output. The profile
// keys on this so the two questions are answered by one row.
const char * gk_cuda_mm_last_path(void);
void gk_cuda_fp4_stats(double * sq_err, double * sq_ref,
                       unsigned long long * zero_groups, unsigned long long * groups);
void gk_cuda_mul_mat_id(gkStream_t stream, struct gk_tensor * dst);

// Launches the fused (rms_norm, mul) pair the backend's fusion plan
// approved: one kernel, writing only the mul's destination.
void gk_cuda_fused_rms_mul(gkStream_t stream, const struct gk_tensor * norm,
                           const struct gk_tensor * mul);

// The three-op residual chain (add, rms_norm, mul): one kernel writing both
// the add's and the mul's destinations.
// The tail fusions: same contracts as the pair above, but the parts need not
// be adjacent - the plan proved the gap safe. All are bit-exact against the
// chains they replace.
void gk_cuda_fused_rms_mul_x(gkStream_t stream, const struct gk_tensor * norm,
                             const struct gk_tensor * mul);
void gk_cuda_fused_add_rms_mul_x(gkStream_t stream, const struct gk_tensor * add,
                                 const struct gk_tensor * norm, const struct gk_tensor * mul);
void gk_cuda_fused_madd(gkStream_t stream, const struct gk_tensor * mul,
                        const struct gk_tensor * add, const struct gk_tensor * add2);
void gk_cuda_fused_unary_mul(gkStream_t stream, const struct gk_tensor * un,
                             const struct gk_tensor * mul);
void gk_cuda_fused_rope_pair(gkStream_t stream, const struct gk_tensor * m1,
                             const struct gk_tensor * m2, const struct gk_tensor * add);

void gk_cuda_fused_add_rms_mul(gkStream_t stream, const struct gk_tensor * add,
                               const struct gk_tensor * norm, const struct gk_tensor * mul);
void gk_cuda_flash_attn(gkStream_t stream, struct gk_cuda_scratch * scratch,
                        struct gk_tensor * dst);
