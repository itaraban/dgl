/**
 *  Copyright (c) 2022 by Contributors
 * @file array/cpu/gather_mm.h
 * @brief GATHER_MM CPU kernel function header.
 */
#ifndef DGL_ARRAY_CPU_GATHER_MM_H_
#define DGL_ARRAY_CPU_GATHER_MM_H_

#include <dgl/array.h>
#include <dgl/bcast.h>
#include <libxsmm.h>

#include <utility>
namespace dgl {
namespace aten {
namespace cpu {

template <typename DType>
void transpose(const DType *in, DType *out, const int N, const int M) {
#pragma omp parallel for
  for (int n = 0; n < N * M; n++) {
    int i = n / N;
    int j = n % N;
    out[n] = in[M * j + i];
  }
}

template <typename DType>
void matmul(
    const DType *A, const DType *B, DType *C, const int M, const int N,
    const int K) {
#pragma omp parallel
  {
    int i, j, k;
#pragma omp for
    for (i = 0; i < M; i++) {
      for (j = 0; j < N; j++) {
        DType local_accum = 0;
        for (k = 0; k < K; k++) {
          local_accum += A[i * K + k] * B[k * N + j];
        }
        C[i * N + j] = local_accum;
      }
    }
  }
}

/**
 * @brief CPU kernel of Gather_mm. The input matrix A is expected to be
 *        sorted according to relation type.
 * @param A The input dense matrix of dimension m x k
 * @param B The input dense matrix of dimension k x n
 * @param C The output dense matrix od dimension m x n
 * @param A_dim1_per_rel The number of rows in each relation in A
 * @param B_dim1_per_rel The number of rows in each relation in B
 * @param a_trans Matrix A to be transposed
 * @param b_trans Matrix B to be transposed
 */
template <int XPU, typename IdType, typename DType>
void gatherMM_SortedEtype(
    const NDArray A, const NDArray B, NDArray C, const NDArray A_dim1_per_rel,
    const NDArray B_dim1_per_rel, bool a_trans, bool b_trans) {
  assert(A_dim1_per_rel.NumElements() == B_dim1_per_rel.NumElements());
  int64_t num_rel = A_dim1_per_rel.NumElements();
  const DType *A_data = A.Ptr<DType>();
  const DType *B_data = B.Ptr<DType>();
  const IdType *A_rel_data = A_dim1_per_rel.Ptr<IdType>();
  const IdType *B_rel_data = B_dim1_per_rel.Ptr<IdType>();
  DType *C_data = C.Ptr<DType>();

  int64_t A_offset = 0, B_offset = 0, C_offset = 0;
  int64_t m, n, k, h_col, w_row;
  for (int etype = 0; etype < num_rel; ++etype) {
    assert(
        (a_trans)                  ? A_rel_data[etype]
        : A->shape[1] == (b_trans) ? B->shape[1]
                                   : B_rel_data[etype]);
    m = A_rel_data[etype];  // rows of A
    n = B->shape[1];        // cols of B
    k = B_rel_data[etype];  // rows of B == cols of A

    NDArray A_trans, B_trans;
    if (a_trans) {
      A_trans = NDArray::Empty({m * k}, A->dtype, A->ctx);
      transpose<DType>(
          A_data + A_offset, static_cast<DType *>(A_trans->data), m, k);
    }
    if (b_trans) {
      B_trans = NDArray::Empty({k * n}, B->dtype, B->ctx);
      transpose<DType>(
          B_data + B_offset, static_cast<DType *>(B_trans->data), k, n);
    }
    if (a_trans || b_trans) {
      int64_t tmp = k;
      if (a_trans) std::swap(m, k);
      if (b_trans) {
        k = tmp;
        std::swap(n, k);
      }
    }
    matmul<DType>(
        (a_trans) ? static_cast<DType *>(A_trans->data) : A_data + A_offset,
        (b_trans) ? static_cast<DType *>(B_trans->data) : B_data + B_offset,
        C_data + C_offset, m, n, k);
    A_offset += m * k;
    B_offset += k * n;
    C_offset += m * n;
  }
}

template <typename DType>
void MMLibxsmm(
    char transA, char transB, int64_t m, int64_t n, int64_t k, DType alpha,
    const DType *A, int64_t lda, const DType *B, int64_t ldb, DType beta,
    DType *C, int64_t ldc) {
  const int flags_trans = LIBXSMM_GEMM_FLAGS(transA, transB);
  const int flags_ab = (LIBXSMM_NEQ(0, beta) ? 0 : LIBXSMM_GEMM_FLAG_BETA_0);
  libxsmm_datatype datatype = LIBXSMM_DATATYPE_UNSUPPORTED;
  if (std::is_same<DType, double>::value) datatype = LIBXSMM_DATATYPE_F64;
  if (std::is_same<DType, float>::value) datatype = LIBXSMM_DATATYPE_F32;
  if (std::is_same<DType, BFloat16>::value) datatype = LIBXSMM_DATATYPE_BF16;
  const libxsmm_gemm_shape gemm_shape = libxsmm_create_gemm_shape(
      m, n, k, lda, ldb, ldc, datatype, datatype, datatype, datatype);

  const libxsmm_gemmfunction kernel = libxsmm_dispatch_gemm_v2(
      gemm_shape, (libxsmm_bitfield)(flags_trans | flags_ab),
      (libxsmm_bitfield)LIBXSMM_GEMM_PREFETCH_NONE);

  assert(NULL != kernel);

  libxsmm_gemm_param gemm_param;
  gemm_param.c.primary = C;
  gemm_param.a.primary = (DType *)(A);
  gemm_param.b.primary = (DType *)(B);
  kernel(&gemm_param);
}

template <typename IdType, typename DType>
void SegmentMM(
    const NDArray A, const NDArray B, NDArray C, const NDArray seglen_A,
    bool a_trans, bool b_trans) {
  const DType *A_data = A.Ptr<DType>();
  const DType *B_data = B.Ptr<DType>();
  const IdType *seglen_A_data = seglen_A.Ptr<IdType>();
  DType *C_data = C.Ptr<DType>();
  int64_t A_offset = 0, B_offset = 0, C_offset = 0;
  int64_t m, n, k;
  int64_t num_rel = seglen_A.NumElements();
  DType alpha = 1., beta = 0.;

  n = B->shape[2];
  k = B->shape[1];
  int ldb = n, lda = k, ldc = n;

  char transB = 'n';
  char transA = 'n';
  if (b_trans) {
    transB = 't';
    ldb = n, lda = n, ldc = k;
    std::swap(n, k);
  }

  for (IdType etype = 0; etype < num_rel; ++etype) {
    m = seglen_A_data[etype];
    MMLibxsmm(
        transB, transA, n, m, k, alpha, B_data + B_offset, ldb,
        A_data + A_offset, lda, beta, C_data + C_offset, ldc);

    A_offset += m * k;
    B_offset += k * n;
    C_offset += m * n;
  }
}

template <typename IdType, typename DType>
void SegmentMMBackwardB(
    const NDArray A, const NDArray dC, NDArray dB, const NDArray seglen) {
  const DType *A_data = A.Ptr<DType>();
  const DType *dC_data = dC.Ptr<DType>();
  const IdType *seglen_data = seglen.Ptr<IdType>();
  DType *dB_data = dB.Ptr<DType>();
  int64_t A_offset = 0, dC_offset = 0, dB_offset = 0;
  int64_t m, n, k;
  int64_t num_rel = seglen.NumElements();
  DType alpha = 1., beta = 0.;

  m = dC->shape[1];
  n = A->shape[1];
  int lddC = m, ldA = n, lddB = m;

  char trans_dC = 'n';
  char trans_A = 't';

  for (IdType etype = 0; etype < num_rel; ++etype) {
    k = seglen_data[etype];

    MMLibxsmm(
        trans_dC, trans_A, m, n, k, alpha, dC_data + dC_offset, lddC,
        A_data + A_offset, ldA, beta, dB_data + dB_offset, lddB);

    dC_offset += m * k;
    A_offset += n * k;
    dB_offset += m * n;
  }
}

}  // namespace cpu
}  // namespace aten
}  // namespace dgl

#endif  // DGL_ARRAY_CPU_GATHER_MM_H_
