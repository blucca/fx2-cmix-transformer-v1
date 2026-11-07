// Original twelve-layer native model, two complete causal pieces, local MLP caps.
#include "model.h"
#include "model_shape.h"
#include "weights_io.h"

#include <array>
#include <chrono>
#include <cmath>
#include <cstdarg>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <string>
#include <sys/resource.h>
#include <vector>

#ifdef __FAST_MATH__
#error "native MLP capture requires strict floating point"
#endif

namespace {
constexpr int NL = fx2::shape::NL, D = fx2::shape::D;
constexpr int DM = fx2::shape::DMLP, V = fx2::shape::V;
static_assert(NL == 12 && D == 192 && DM == 768 && V == 205);
static_assert(sizeof(float) == 4 && sizeof(uint16_t) == 2);
using Clock = std::chrono::steady_clock;

[[noreturn]] void Fail(const char* fmt, ...) {
  va_list ap;
  va_start(ap, fmt);
  std::fprintf(stderr, "mlp-capture: ");
  std::vfprintf(stderr, fmt, ap);
  std::fprintf(stderr, "\n");
  va_end(ap);
  std::exit(1);
}

FILE* Open(const std::string& path, const char* mode) {
  FILE* f = std::fopen(path.c_str(), mode);
  if (!f) Fail("open failed: %s", path.c_str());
  return f;
}

void Close(FILE* f) {
  const bool bad = std::ferror(f);
  const int result = std::fclose(f);
  if (bad || result) Fail("file I/O failed");
}

void WriteF32(FILE* f, const float* x, size_t n) {
  if (std::fwrite(x, sizeof(float), n, f) != n) Fail("F32 write failed");
}

template<class T>
std::vector<T> Read(const std::string& path, size_t count) {
  FILE* f = Open(path, "rb");
  std::vector<T> x(count);
  if (std::fread(x.data(), sizeof(T), count, f) != count ||
      std::fgetc(f) != EOF) Fail("input size mismatch: %s", path.c_str());
  Close(f);
  return x;
}

void MakeDirectory(const std::string& path) {
  if (!std::filesystem::create_directory(path))
    Fail("fresh output directory required: %s", path.c_str());
}

double CpuSeconds() {
  rusage u{};
  if (getrusage(RUSAGE_SELF, &u)) Fail("getrusage failed");
  return u.ru_utime.tv_sec + u.ru_utime.tv_usec * 1e-6 +
         u.ru_stime.tv_sec + u.ru_stime.tv_usec * 1e-6;
}

double Seconds(Clock::time_point start) {
  return std::chrono::duration<double>(Clock::now() - start).count();
}

void DumpMlp(const char* weights, const std::string& input,
             const std::string& output) {
  const auto expected = Read<uint32_t>(input + "/mlp-qhist.u32", NL * 2 * 15);
  const auto wf = fx2::WeightsFile::load_compressed(weights);
  if (wf.tensors.size() != 434) Fail("original tensor count mismatch");
  FILE* matrix = Open(output + "/original-matrices.f32", "wb");
  FILE* meta = Open(output + "/original-matrices.json", "w");
  std::fprintf(meta, "[\n");
  size_t offset = 0;
  for (int l = 0; l < NL; l++) for (int kind = 0; kind < 2; kind++) {
    const char* which = kind == 0 ? "up" : "down";
    const int dout = kind == 0 ? DM : D, din = kind == 0 ? D : DM;
    const std::string prefix = "blocks." + std::to_string(l) + ".mlp." + which;
    const auto& q = wf.get(prefix + ".weight.q", fx2::DT_I8,
                          {uint32_t(dout), uint32_t(din)});
    const auto& s = wf.get(prefix + ".weight.scale", fx2::DT_BF16,
                          {uint32_t(dout)});
    const auto& a = wf.get(prefix + ".quantize_activation.scale", fx2::DT_BF16, {1});
    const float act = fx2::bf16_to_f32(a.bf16_bits()[0]);
    if (!std::isfinite(act) || !(act > 0.0f)) Fail("activation scale failed");
    std::array<size_t, 15> hist{};
    std::vector<float> row_scales(dout), w(q.numel);
    for (int o = 0; o < dout; o++) {
      row_scales[o] = fx2::bf16_to_f32(s.bf16_bits()[o]);
      if (!std::isfinite(row_scales[o]) || row_scales[o] == 0.0f)
        Fail("row scale failed: %s row %d", prefix.c_str(), o);
    }
    for (size_t i = 0; i < q.numel; i++) {
      const int v = q.i8()[i];
      if (v < -7 || v > 7) Fail("weight INT4 range failed");
      hist[v + 7]++;
      w[i] = static_cast<float>(v) * row_scales[i / din];
    }
    for (int v = 0; v < 15; v++)
      if (hist[v] != expected[(l * 2 + kind) * 15 + v])
        Fail("public q histogram mismatch: %s bin %d", prefix.c_str(), v);
    WriteF32(matrix, w.data(), w.size());
    std::fprintf(meta, "%s{\"name\":\"%s.weight.q\",\"shape\":[%d,%d],"
        "\"elements\":%zu,\"offset_bytes\":%zu,\"matrix_file\":\"original-matrices.f32\","
        "\"activation_scale_f32\":%.9g,\"activation_scale_bf16_bits\":%u,\"histogram\":[",
        offset ? ",\n" : "", prefix.c_str(), dout, din, w.size(), offset,
        double(act), unsigned(a.bf16_bits()[0]));
    for (int v = 0; v < 15; v++) std::fprintf(meta, "%s%zu", v ? "," : "", hist[v]);
    std::fprintf(meta, "],\"row_scales_f32\":[");
    for (int o = 0; o < dout; o++)
      std::fprintf(meta, "%s%.9g", o ? "," : "", double(row_scales[o]));
    std::fprintf(meta, "],\"row_scale_bf16_bits\":[");
    for (int o = 0; o < dout; o++)
      std::fprintf(meta, "%s%u", o ? "," : "", unsigned(s.bf16_bits()[o]));
    std::fprintf(meta, "]}");
    offset += w.size() * sizeof(float);
  }
  std::fprintf(meta, "\n]\n");
  Close(matrix);
  Close(meta);
}

// Every site stores raw signed INT8 coordinates as exactly representable F32.
constexpr const char* SITES[4] = {
    "mlp_up_input_int8", "mlp_up_out", "mlp_down_input_int8", "mlp_out"};
constexpr int WIDTHS[4] = {D, DM, DM, D};

std::string CaptureName(int layer, int site) {
  char name[64];
  std::snprintf(name, sizeof name, "%02d_%s.f32", layer, SITES[site]);
  return name;
}

void RunPiece(fx2::Transformer& model, const char* label, size_t rows,
              const std::string& input, const std::string& output, FILE* index) {
  const auto start = Clock::now();
  const double cpu = CpuSeconds();
  const std::string dir = output + "/" + label;
  MakeDirectory(dir);
  const auto tokens = Read<uint8_t>(input + "/" + label + "-input.u8", rows);
  const auto targets = Read<uint8_t>(input + "/" + label + "-target.u8", rows);
  const auto priors = Read<uint16_t>(input + "/" + label + "-priors.f16", rows * V);
  FILE* files[NL][4]{};
  size_t counts[NL][4]{};
  for (int l = 0; l < NL; l++) for (int site = 0; site < 4; site++)
    files[l][site] = Open(dir + "/" + CaptureName(l, site), "wb");
  FILE* logits_out = Open(dir + "/12_logits.f32", "wb");
  model.set_capture([&](const char* name, const float* values, int n) {
    if (name[0] < '0' || name[0] > '9' || name[1] < '0' ||
        name[1] > '9' || name[2] != '_') return;
    const int l = (name[0] - '0') * 10 + name[1] - '0';
    if (l >= NL) return;
    for (int site = 0; site < 4; site++) if (std::strcmp(name + 3, SITES[site]) == 0) {
      if (n != WIDTHS[site]) Fail("capture width failed: %s", name);
      WriteF32(files[l][site], values, size_t(n));
      counts[l][site]++;
      return;
    }
  });
  model.begin_article(0);
  double nll_probs = 0.0, nll_logits = 0.0;
  float probs[V];
  for (size_t r = 0; r < rows; r++) {
    if (targets[r] >= V) Fail("target range failed");
    model.step(tokens[r], priors.data() + r * V, probs);
    const float* logits = model.last_logits();
    WriteF32(logits_out, logits, V);
    const float p = probs[targets[r]];
    if (!std::isfinite(p) || !(p > 0.0f)) Fail("native target probability failed");
    nll_probs -= std::log(double(p));
    double maximum = logits[0], denominator = 0.0;
    for (int v = 0; v < V; v++) {
      if (!std::isfinite(logits[v])) Fail("logit finite check failed");
      if (logits[v] > maximum) maximum = logits[v];
    }
    for (int v = 0; v < V; v++) denominator += std::exp(double(logits[v]) - maximum);
    nll_logits += maximum + std::log(denominator) - logits[targets[r]];
  }
  model.set_capture({});
  Close(logits_out);
  for (int l = 0; l < NL; l++) for (int site = 0; site < 4; site++) {
    if (counts[l][site] != rows) Fail("capture row count failed");
    Close(files[l][site]);
  }
  const double wall = Seconds(start), used_cpu = CpuSeconds() - cpu;
  std::fprintf(index, "{\"name\":\"%s\",\"rows\":%zu,\"rope_position_offset\":0,"
      "\"begin_article_calls\":1,\"step_calls\":%zu,\"last_logits_f32\":\"%s/12_logits.f32\","
      "\"nll_sum_from_probs\":%.17g,\"mean_nll_from_probs\":%.17g,"
      "\"nll_sum_from_logits_f64\":%.17g,\"mean_nll_from_logits_f64\":%.17g,"
      "\"wall_seconds\":%.9g,\"cpu_seconds\":%.9g,\"graphs\":[",
      label, rows, rows, label, nll_probs, nll_probs / rows,
      nll_logits, nll_logits / rows, wall, used_cpu);
  for (int l = 0; l < NL; l++) for (int kind = 0; kind < 2; kind++) {
    const int in_site = kind * 2, out_site = in_site + 1;
    std::fprintf(index, "%s{\"name\":\"blocks.%d.mlp.%s.weight.q\",\"rows\":%zu,"
        "\"input_i8\":\"%s/%s\",\"input_storage_dtype\":\"float32le\","
        "\"input_columns\":%d,\"output_f32\":\"%s/%s\",\"output_columns\":%d,"
        "\"input_capture_rows\":%zu,\"output_capture_rows\":%zu}",
        l || kind ? "," : "", l, kind == 0 ? "up" : "down", rows,
        label, CaptureName(l, in_site).c_str(), WIDTHS[in_site],
        label, CaptureName(l, out_site).c_str(), WIDTHS[out_site],
        counts[l][in_site], counts[l][out_site]);
  }
  std::fprintf(index, "]}");
  std::printf("CAPTURE_PIECE %s rows=%zu native_nll=%.9g cpu_seconds=%.6f\n",
              label, rows, nll_probs / rows, used_cpu);
  std::fflush(stdout);
}
}  // namespace

int main(int argc, char** argv) {
  if (argc != 4) Fail("usage: capture-main WEIGHTS_TFWC2 INPUT_DIR FRESH_OUTPUT_DIR");
  const auto start = Clock::now();
  const double cpu = CpuSeconds();
  const std::string input = argv[2], output = argv[3];
  MakeDirectory(output);
  DumpMlp(argv[1], input, output);
  fx2::Transformer model(argv[1]);
  FILE* index = Open(output + "/capture-index.json", "w");
  std::fprintf(index, "{\"schema\":\"hutter.native-mlp-capture.round455.v1\","
      "\"status\":\"passed\",\"layers\":12,\"graphs\":24,\"weight_tensors\":434,"
      "\"weights_metadata\":\"original-matrices.json\",\"weights_f32\":\"original-matrices.f32\","
      "\"input_units\":\"signed INT8 coordinates stored as F32; X = activation_scale_f32 * q8\","
      "\"model_instances\":1,\"compressed_package_reads\":2,\"pieces\":[");
  RunPiece(model, "train", 1199, input, output, index);
  std::fprintf(index, ",");
  RunPiece(model, "held", 1606, input, output, index);
  rusage u{};
  if (getrusage(RUSAGE_SELF, &u)) Fail("getrusage failed");
  std::fprintf(index, "],\"resource_scope\":\"capture process including both weight decodes; compiler excluded\","
      "\"cpu_seconds\":%.9g,\"wall_seconds\":%.9g,\"peak_rss_kib\":%ld}\n",
      CpuSeconds() - cpu, Seconds(start), u.ru_maxrss);
  Close(index);
  std::printf("CAPTURE_COMPLETE output=%s\n", output.c_str());
  return 0;
}
