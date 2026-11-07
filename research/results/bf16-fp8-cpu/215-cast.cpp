// Generated kernel body; the Python controls create its build prefix and bindings.
extern "C"  void  kernel(const float* in_ptr0,
                       float* out_ptr0)
{
    std::atomic<int> inductor_cpu_integer_div_error{0};
    inductor_cpu_integer_div_error_flag = &inductor_cpu_integer_div_error;
    {
        #pragma GCC ivdep
        for(int64_t x0=static_cast<int64_t>(0L); x0<static_cast<int64_t>(65536L); x0+=static_cast<int64_t>(1L))
        {
            {
                {
                    auto tmp0 = in_ptr0[static_cast<int64_t>(x0)];
                    auto tmp1 = c10::convert<at::BFloat16>(tmp0);
                    auto tmp2 = c10::convert<at::Float8_e4m3fn>(tmp1);
                    auto tmp3 = c10::convert<float>(tmp2);
                    out_ptr0[static_cast<int64_t>(x0)] = tmp3;
                }
            }
        }
    }
    inductor_cpu_integer_div_error_flag = nullptr;
    inductor_cpu_throw_if_integer_div_error(inductor_cpu_integer_div_error);
}

