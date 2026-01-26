# CUDA-Matrix-Optimization
# Optimized Matrix Multiplication on NVIDIA Ampere (RTX 3070)

**Objective:** Designed a high-performance custom CUDA kernel to compete with NVIDIA's cuBLAS library for SGEMM (Single Precision General Matrix Multiply).

**Techniques Used:**
->Shared Memory Tiling: Reduced Global Memory bandwidth consumption by 16x.

->Coalesced Memory Access: Optimized warp execution to ensure 100% bus utilization.

->Bank Conflict Avoidance: Structured shared memory access patterns to minimize serialization.


**Performance Results (2048x2048 Matrix):**

->Naive CPU Implementation: ~1500 ms (Estimated)

->My Tiled CUDA Kernel: 14.46 ms

->NVIDIA cuBLAS: 8.14 ms


**Result:** Achieved 56% of theoretical peak performance relative to the vendor library.
