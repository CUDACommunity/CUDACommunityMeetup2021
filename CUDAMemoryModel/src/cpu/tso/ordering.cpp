#include <thread>
#include <atomic>
#include <iostream>

#include "thread_body.h"

class barrier_class
{
  unsigned int threads_count {};
  std::atomic<unsigned int> threads_at_epoch;
  std::atomic<unsigned int> epoch;
public:
  barrier_class () = delete;
  barrier_class (unsigned int threads_count_arg) 
    : threads_count (threads_count_arg) 
    , threads_at_epoch (0)
    , epoch (0)
  {}

  void operator() ()
  {
    const unsigned int current_epoch = epoch.load ();

    if (threads_at_epoch.fetch_add (1) == threads_count - 1)
      {
        threads_at_epoch = 0;
        epoch++;
      }
    else while (epoch.load () == current_epoch);
  }
};

template <typename action_type>
void thread_body (int iterations_count, barrier_class &barrier, const action_type &action)
{
  for (int i = 0; i < iterations_count; i++)
    {
      barrier ();
      action ();
      barrier ();
    }
}

int main()
{
  const int max_iterations_count = 100'000;
  barrier_class barrier (3);

  std::thread thread_1 ([&] { thread_body (max_iterations_count, barrier, thread_1_body); });
  std::thread thread_2 ([&] { thread_body (max_iterations_count, barrier, thread_2_body); });

  int violations_count = 0;

  // Repeat the experiment ad infinitum
  for (int iteration = 0; iteration < max_iterations_count; iteration++)
    {
      reset_reg ();

      barrier (); ///< Let threads execute the body
      barrier (); ///< Wait for both threads to complete

      if (check_for_reordering ())
        violations_count++;
    }

  if (violations_count)
    std::cout << violations_count << "\n";
  else
    std::cout << "ok\n";

  thread_1.join ();
  thread_2.join ();
  return 0;  // Never returns
}

