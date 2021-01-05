int X, Y;

#define USE_COMPILER_BARRIER 1

void foo()
{
  X = Y + 1;
#if USE_COMPILER_BARRIER
  asm volatile ("" ::: "memory");
#endif
  Y = 0;
}
