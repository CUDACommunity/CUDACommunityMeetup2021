#include "thread_body.h"

#define USE_CPU_FENCE 0
#define USE_COMPILER_FENCE 0

int X, Y;
int r1, r2;

void thread_1_body ()
{
  X = 1;
#if USE_CPU_FENCE
  asm volatile("mfence" ::: "memory");  // Prevent CPU reordering
#elif USE_COMPILER_FENCE
  asm volatile("" ::: "memory");  // Prevent compiler reordering
#endif
  r1 = Y;
}

void thread_2_body ()
{
  Y = 1;
#if USE_CPU_FENCE
  asm volatile("mfence" ::: "memory");  // Prevent CPU reordering
#elif USE_COMPILER_FENCE
  asm volatile("" ::: "memory");  // Prevent compiler reordering
#endif
  r2 = X;
}

bool check_for_reordering ()
{
  return r1 == 0 && r2 == 0;
}

void reset_reg ()
{
  X = 0;
  Y = 0;
}
