
#define USE_COMPILER_BARRIER 1

__global__ void foo (int *X, int n)
{
  for (int i = 0; i < 100; i++)
  {
    X[i] *= n;
#if USE_COMPILER_BARRIER
    // __threadfence ();
    asm volatile ("" ::: "memory");
#endif
  }
}
