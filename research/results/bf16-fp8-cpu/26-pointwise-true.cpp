// Generated kernel body; the Python controls create its build prefix and bindings.
extern "C"  void kernel(const float* in_ptr0,
                       float* out_ptr0)
{
    {
        #pragma GCC ivdep
        for(int64_t x0=static_cast<int64_t>(0L); x0<static_cast<int64_t>(65536L); x0+=static_cast<int64_t>(1L))
        {
            {
                {
                    auto tmp0 = in_ptr0[static_cast<int64_t>(x0)];
                    auto tmp1 = c10::convert<float>(tmp0);
                    auto tmp2 = static_cast<float>(1.5);
                    auto tmp3 = decltype(tmp1)(tmp1 * tmp2);
                    auto tmp4 = c10::convert<float>(tmp3);
                    auto tmp5 = c10::convert<float8_e4m3fn>(tmp4);
                    auto tmp6 = c10::convert<float>(tmp5);
                    out_ptr0[static_cast<int64_t>(x0)] = tmp6;
                }
            }
        }
    }
}

