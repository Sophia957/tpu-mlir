//===----------------------------------------------------------------------===//
//
// Copyright (C) 2022 Sophgo Technologies Inc.  All rights reserved.
//
// TPU-MLIR is licensed under the 2-Clause BSD License except for the
// third-party components.
//
//===----------------------------------------------------------------------===//

#include "tpu_mlir/Support/MathUtils.h"
#include "float.h"
#include "mlir/IR/PatternMatch.h"
#include "omp.h"
#include "tpu_mlir/Support/Helper/Quant.h"
#include <map>
#include "tpu_mlir/Support/Dnnl/Dnnl.h"

namespace tpu_mlir {

void get_scale_and_shift(float scale_f, int &scale, int &shift, int bitwidth) {
  float min_err = FLT_MAX;
  int m_limit = (bitwidth == 32) ? INT_MAX : CHAR_MAX;
  for (int n = -32; n < 31; n++) {
    //若scale_f大于等于1，这里循环上限要设为31(而不是32)，且越大则需减少越多，暂只考虑scale_f小于等于1的情形
    // wxc 20220119
    int m = (int)std::round(scale_f * std::pow(2, n));
    float err = std::abs(m / std::pow(2, n) - scale_f);
    if (err < min_err && abs(m) < m_limit) {
      min_err = err;
      shift = n;
    }
  }
  scale = (int)std::round(scale_f * std::pow(2, shift));
}

void get_scale_and_shift_positive(float scale_f, int &scale, int &shift,
                                  int bitwidth) {
  float min_err = FLT_MAX;
  int m_limit = (bitwidth == 32) ? INT_MAX : CHAR_MAX;
  for (int n = 0; n < 31; n++) {
    int m = (int)std::round(scale_f * std::pow(2, n));
    float err = std::abs(m / std::pow(2, n) - scale_f);
    if (err < min_err && abs(m) < m_limit) {
      min_err = err;
      shift = n;
    }
  }
  scale = (int)std::round(scale_f * std::pow(2, shift));
}

// this function search positive right shift with max_shift, max_shift set to 8
// for int16 op and shift to 8bit output.
void get_scale_and_shift_positive_maxshift(float scale_f, int &scale,
                                           int &shift, int bitwidth,
                                           int max_shift) {
  float min_err = FLT_MAX;
  int m_limit = (bitwidth == 32) ? INT_MAX : CHAR_MAX;
  for (int n = 0; n < max_shift; n++) {
    int m = (int)std::round(scale_f * std::pow(2, n));
    float err = std::abs(m / std::pow(2, n) - scale_f);
    if (err < min_err && abs(m) < m_limit) {
      min_err = err;
      shift = n;
    }
  }
  scale = (int)std::round(scale_f * std::pow(2, shift));
}

template <typename Dtype> float findMaxabs(const Dtype *pSrcData, int len) {
  float fmax = 0.0;
  float dataTmp;
  for (int i = 0; i < len; i++) {
    dataTmp = (float)pSrcData[i];
    fmax = (fabs(dataTmp) > fmax) ? fabs(dataTmp) : fmax;
  }
  if (fmax == 0.0) {
    ; // LOG(WARNING) << "findMaxabs meet fmax == 0";
  }

  return fmax;
}
template float findMaxabs<float>(const float *pSrcData, int len);
template float findMaxabs<int>(const int *pSrcData, int len);

template <typename Dtype>
void findMinMax(const Dtype *pSrcData, int len, Dtype *minVal, Dtype *maxVal) {
  Dtype fmin = pSrcData[0];
  Dtype fmax = pSrcData[0];
  Dtype dataTmp;
  for (int i = 0; i < len; i++) {
    dataTmp = pSrcData[i];
    fmin = dataTmp < fmin ? dataTmp : fmin;
    fmax = dataTmp > fmax ? dataTmp : fmax;
  }
  *minVal = fmin;
  *maxVal = fmax;
}

template void findMinMax<float>(const float *pSrcData, int len, float *minVal,
                                float *maxVal);
template void findMinMax<int>(const int *pSrcData, int len, int *minVal,
                              int *maxVal);

int calRightShiftNum(float fmax, double thBottom, double thTop, int numBits) {
  double dataTmp = 1.0 * fmax * thBottom / thTop;
  int m = 0;

  if (dataTmp <= 0.0) {
    llvm::errs() << "meet dataTmp <= 0.0, fmax:" << fmax
                 << " thBottom:" << thBottom << " thTop:" << thTop;
    return 0;
  }
  while (dataTmp < ((1 << (numBits - 1)) - 1)) {
    dataTmp = dataTmp * 2;
    m = m + 1;
  }

  m = m > 32 ? 31 : m - 1;
  return m;
}

template <typename T> void func_abs(int n, T *src, T *dst) {
  for (int i = 0; i < n; i++) {
    dst[i] = std::abs(src[i]);
  }
}

template <typename T> void func_log(int n, T *src, T *dst) {
  for (int i = 0; i < n; i++) {
    dst[i] = std::log(src[i]);
  }
}

int calRightShiftNumUseCblas(float fmax, double thBottom, double thTop,
                             int numBits) {
  func_abs(1, &fmax, &fmax);
  double dataTmp = 1.0 * ((1 << (numBits - 1)) - 1) / (fmax * thBottom / thTop);
  int m = 0;

  double log_dem, log_num;
  double data2 = 2.0;

  func_log(1, &dataTmp, &log_dem);
  func_log(1, &data2, &log_num);

  m = floor(log_dem / log_num);
  m = m > 31 ? 31 : m;

  return m;
}

float func_log2(double dataInput) {
  double log_dem, log_num;
  double data2 = 2.0;
  float result;

  func_log(1, &dataInput, &log_dem);
  func_log(1, &data2, &log_num);

  result = log_dem / log_num;

  return result;
}

float quantizeToInt16(const float *pSrc, int16_t *pDst, int len, float scale,
                      int rshift) {
  int16_t qmax = 32767;
  int16_t qmin = -32768;
  int tmp = 0;
  int overflow = 0;
  int half_data = (rshift == 0) ? 0 : 1 << (rshift - 1);

  for (int i = 0; i < len; i++) {
    tmp = round(pSrc[i] * scale);
    tmp = (tmp + half_data) >> rshift;
    pDst[i] = (int16_t)((tmp > qmax) ? qmax : ((tmp < qmin) ? qmin : tmp));
    overflow = (tmp > qmax || tmp < qmin) ? overflow + 1 : overflow;
  }
  float ratio = ((float)overflow) / ((float)len);
  if (ratio > 0 && len > 1) {
    llvm::errs() << "ratio of overflow = " << ratio;
  }
  return ratio;
}

float quantizeToInt15(const float *pSrc, int16_t *pDst, int len, float scale,
                      int rshift) {
  int16_t qmax = 16383;
  int16_t qmin = -16384;
  int tmp = 0;
  int overflow = 0;
  int half_data = (rshift == 0) ? 0 : 1 << (rshift - 1);

  for (int i = 0; i < len; i++) {
    tmp = floor(pSrc[i] * scale + 0.5);
    tmp = (tmp + half_data) >> rshift;
    pDst[i] = (int16_t)((tmp > qmax) ? qmax : ((tmp < qmin) ? qmin : tmp));
    overflow = (tmp > qmax || tmp < qmin) ? overflow + 1 : overflow;
  }
  float ratio = ((float)overflow) / ((float)len);
  if (ratio > 0) {
    llvm::errs() << "ratio of overflow = " << ratio;
  }
  return ratio;
}

void quantizeToInt8(const float *pSrc, int8_t *pDst, int len, float scale) {
  int8_t qmax = 127;
  int8_t qmin = -128;
  int tmp = 0;

  for (int i = 0; i < len; i++) {
    tmp = round(pSrc[i] * scale);
    pDst[i] = (int8_t)((tmp > qmax) ? qmax : ((tmp < qmin) ? qmin : tmp));
  }
}

// tensorflow/lite/kernels/internal/quantization_util.cc
// mlir/lib/Dialect/Tosa/Utils/QuantUtils.cpp
// to compitable with tflite
void QuantizeMultiplier(double double_multiplier, int64_t *quantized_multiplier,
                        int64_t *shift) {
  if (double_multiplier == 0.) {
    *quantized_multiplier = 0;
    *shift = 0;
    return;
  }
  int shift_tmp;
  const double q = std::frexp(double_multiplier, &shift_tmp);
  *shift = shift_tmp;
  auto q_fixed = static_cast<int64_t>(std::round(q * (1LL << 31)));
  assert(q_fixed <= (1LL << 31));
  if (q_fixed == (1LL << 31)) {
    q_fixed /= 2;
    ++*shift;
  }
  assert(q_fixed <= std::numeric_limits<int32_t>::max());
  // A shift amount smaller than -31 would cause all bits to be shifted out
  // and thus all results would be zero. We implement that instead with
  // q_fixed==0, so as to avoid hitting issues with right-shift
  // operations with shift amounts greater than 31. Note that this happens
  // roughly when abs(double_multiplier) < 2^-31 and the present handling means
  // that we're effectively flushing tiny double_multiplier's to zero.
  // We could conceivably handle values in the range (roughly) [32, 63]
  // as 'denormals' i.e. (shift==0, q_fixed < 2^30). In that point of view
  // the present handling is just doing 'flush denormals to zero'. We could
  // reconsider and actually generate nonzero denormals if a need arises.
  if (*shift < -31) {
    *shift = 0;
    q_fixed = 0;
  }
  // Single-rounding MultiplyByQuantizedMultiplier doesn't support a shift > 30,
  // saturate it.
  if (*shift > 30) {
    *shift = 30;
    q_fixed = (1LL << 31) - 1;
  }
  // Sophgo expects right shift to be positive, and embed (1 << 31) into right
  // shift bits.
  // *shift = (-*shift) + 31;
  *quantized_multiplier = static_cast<int32_t>(q_fixed);
}

template <typename T>
T RightShiftRound(T src, int shift_num, RoundingMode round_mode) {
  if (shift_num == 0)
    return src;
  if (shift_num > 63)
    shift_num = 63;
  T val, res;
  val = src >> shift_num;
  res = val;
  T lo_mask = (1ull << shift_num) - 1;
  T mant = src & lo_mask;
  T mant_0d5 = 1ull << (shift_num - 1);
  if (round_mode == ROUNDING_HALF_TO_EVEN) {
    if (mant == mant_0d5)
      res = val + (val & 1);
    else if (mant > mant_0d5)
      res = val + 1;
  } else if (round_mode == ROUNDING_HALF_AWAY_FROM_ZERO) {
    if (src >= 0 && mant >= mant_0d5)
      res = val + 1;
    else if (src < 0 && mant > mant_0d5)
      res = val + 1;
  } else if (round_mode == ROUNDING_TOWARDS_ZERO) {
    if (src < 0)
      res = val + (mant != 0);
  } else if (round_mode == ROUNDING_DOWN)
    res = val;
  else if (round_mode == ROUNDING_UP)
    res = val + (mant != 0);
  else if (round_mode == ROUNDING_HALF_UP) {
    if (mant >= mant_0d5)
      res = val + 1;
  } else if (round_mode == ROUNDING_HALF_DOWN) {
    if (mant > mant_0d5)
      res = val + 1;
  }
  return res;
}

template long long RightShiftRound(long long src, int shift_num, RoundingMode round_mode);
template int64_t RightShiftRound(int64_t src, int shift_num, RoundingMode round_mode);

// to compilable with tflite
// tensorflow/lite/kernels/internal/common.h:MultiplyByQuantizedMultiplier()
int32_t MultiplyByQuantizedMultiplier(int32_t x, int32_t multiplier,
                                      int shift) {
  // int shift = -(rshift - 31);
  int64_t value = shift > 0 ? x << shift : x;
  value = RightShiftRound(value * multiplier, 31, ROUNDING_HALF_UP);
  if (value > (1ll << 31) - 1)
    value = (1ll << 31) - 1;
  else if (value < -(1ll << 31))
    value = -(1ll << 31);
  if (shift < 0) {
    value = RightShiftRound(value, -shift, ROUNDING_HALF_AWAY_FROM_ZERO);
  }
  return (int32_t)value;
}

void pad_tensor(float *p_after_pad, float *src, int n, int c, int h, int w,
                int pt, int pb, int pl, int pr, float pad_value) {
  int nc = n * c;
  int oh = h + pt + pb;
  int ow = w + pl + pr;
  for (int i = 0; i < nc; i++) {
    for (int j = 0; j < oh; j++) {
      for (int k = 0; k < ow; k++) {
        int d_offset = (i * oh + j) * ow + k;
        if (j < pt || j >= (pt + h) || k < pl || k >= (pl + w)) {
          p_after_pad[d_offset] = pad_value;
        } else {
          int s_offset = (i * h + j - pt) * w + k - pl;
          p_after_pad[d_offset] = src[s_offset];
        }
      }
    }
  }
}

void pad_tensor(float *p_after_pad, float *src, int n, int c, int d, int h,
                int w, int pdf, int pdb, int pht, int phb, int pwl, int pwr,
                float pad_value) {
  int nc = n * c;
  int od = d + pdf + pdb;
  int oh = h + pht + phb;
  int ow = w + pwl + pwr;
  for (int i = 0; i < nc; i++) {
    for (int m = 0; m < od; m++) {
      for (int j = 0; j < oh; j++) {
        for (int k = 0; k < ow; k++) {
          int d_offset = (i * od * oh + m * oh + j) * ow + k;
          if (m < pdf || m >= (pdf + d) || j < pht || j >= (pht + h) ||
              k < pwl || k >= (pwl + w)) {
            p_after_pad[d_offset] = pad_value;
          } else {
            int s_offset = ((i * d + m - pdf) * h + j - pht) * w + k - pwl;
            p_after_pad[d_offset] = src[s_offset];
          }
        }
      }
    }
  }
}

void pad_tensor_for_deconv(float *p_after_pad, float *src, int n, int c, int d,
                           int h, int w, int kd, int kh, int kw, int dd, int dh,
                           int dw, int sd, int sh, int sw, int pdf, int pdb,
                           int pht, int phb, int pwl, int pwr,
                           float pad_value) {
  int nc = n * c;
  int od = (d - 1) * sd + 1 + dd * (kd - 1);
  int oh = (h - 1) * sh + 1 + dh * (kh - 1);
  int ow = (w - 1) * sw + 1 + dw * (kw - 1);
  int pst[3] = {(kd - 1) * dd - pdf, (kh - 1) * dh - pht, (kw - 1) * dw - pwl};
  int ped[3] = {(kd - 1) * dd - pdb, (kh - 1) * dh - phb, (kw - 1) * dw - pwr};
  for (int i = 0; i < nc; i++) {
    for (int m = 0; m < od; m++) {
      for (int j = 0; j < oh; j++) {
        for (int k = 0; k < ow; k++) {
          int d_offset = (i * od * oh + m * oh + j) * ow + k;
          if (m < pst[0] || m >= (od - ped[0]) || j < pst[1] ||
              j >= (oh - ped[1]) || k < pst[2] || k >= (ow - ped[2]) ||
              (m - pst[0]) % sd != 0 || (j - pst[1]) % sh != 0 ||
              (k - pst[2]) % sw != 0) {
            p_after_pad[d_offset] = pad_value;
          } else {
            int s_offset =
                ((i * d + (m - pst[0]) / sd) * h + (j - pst[1]) / sh) * w +
                (k - pst[2]) / sw;
            p_after_pad[d_offset] = src[s_offset];
          }
        }
      }
    }
  }
}


void tensor_sub_zp(float* tensor_after_zp, float* src, int64_t length, float zero_point) {
#pragma omp parallel for schedule(static, omp_schedule(length))
  for (int i = 0; i < length; ++i) {
    tensor_after_zp[i] = src[i] - zero_point;
  }
}

int omp_schedule(int count) {
  return (count + omp_get_num_threads() - 1) / omp_get_num_threads();
}

void function_relu(float *src, float *dst, int64_t size, float relu_limit, mlir::Type elem_type) {
#pragma omp parallel for schedule(static, omp_schedule(size))
  for (int64_t i = 0; i < size; ++i) {
    dst[i] = src[i] > 0 ? src[i] : 0;
    if (relu_limit > 0.f && dst[i] > relu_limit) {
      dst[i] =  relu_limit;
    }
    if (elem_type) {
      if (elem_type.isUnsignedInteger(8)) {
        dst[i] = helper::Quant::to_uint8(dst[i]);
      } else if (elem_type.isInteger(8)) {
        dst[i] = helper::Quant::to_int8(dst[i]);
      }
    }
  }
}

int dnnl_mm(float *input, float *weight, float *bias,
    float *output, int m, int k, int n, bool transpose) {
  if (!bias) {
    auto zero_bias = new std::vector<float>(n, 0.0f);
    bias = zero_bias->data();
  }

#ifdef DUMP_FLAG
  static int dump_idx = 0;
  std::string prefix = std::string("ip") + std::to_string(dump_idx);
  if (dump_idx == 0) {
    write_bianry_file(prefix + std::string("_in.bin"),
        (const char *)input, m * k * sizeof(float));
  }
#endif // DUMP_FLAG

  using tag = memory::format_tag;
  using dt = memory::data_type;

  engine eng(engine::kind::cpu, 0);
  stream s(eng);

  std::vector<primitive> net;
  std::vector<std::unordered_map<int, memory>> net_args;

  memory::dims src_tz = { m, k };
  memory::dims weights_tz = { n, k };
  memory::dims bias_tz = { n };
  memory::dims dst_tz = { m, n };

  if (!bias) {
    auto zero_bias = new std::vector<float>(n, 0.0f);
    bias = zero_bias->data();
  }

  // memory
  auto user_src_memory = memory(
      { { src_tz }, dt::f32, tag::nc }, eng, input);
  auto user_weights_memory = memory(
      { { weights_tz }, dt::f32, tag::oi }, eng, weight);
  auto user_bias_memory = memory(
      { { bias_tz }, dt::f32, tag::x }, eng, bias);
  auto user_dst_memory = memory(
      { { dst_tz }, dt::f32, tag::nc }, eng, output);

  // md
  auto src_md = memory::desc({ src_tz }, dt::f32, tag::any);
  auto weights_md = memory::desc({ weights_tz }, dt::f32, tag::any);
  auto bias_md = memory::desc({ bias_tz }, dt::f32, tag::any);
  auto dst_md = memory::desc({ dst_tz }, dt::f32, tag::any);

  // fc desc
  auto fc_desc = inner_product_forward::desc(prop_kind::forward_inference,
      src_md, weights_md, bias_md, dst_md);
  auto fc_prim_desc = inner_product_forward::primitive_desc(fc_desc, eng);

  // do reorder if needed
  auto src_memory = user_src_memory;
  if (fc_prim_desc.src_desc() != user_src_memory.get_desc()) {
    src_memory = memory(fc_prim_desc.src_desc(), eng);
    net.push_back(reorder(user_src_memory, src_memory));
    net_args.push_back({ { DNNL_ARG_FROM, user_src_memory },
        { DNNL_ARG_TO, src_memory } });
  }
  auto weights_memory = user_weights_memory;
  if (fc_prim_desc.weights_desc() != user_weights_memory.get_desc()) {
    weights_memory = memory(fc_prim_desc.weights_desc(), eng);
    reorder(user_weights_memory, weights_memory)
        .execute(s, user_weights_memory, weights_memory);
  }
  auto bias_memory = user_bias_memory;

  auto dst_memory = memory(fc_prim_desc.dst_desc(), eng);

  net.push_back(inner_product_forward(fc_prim_desc));
  net_args.push_back({ { DNNL_ARG_SRC, src_memory },
      { DNNL_ARG_WEIGHTS, weights_memory },
      { DNNL_ARG_BIAS, bias_memory },
      { DNNL_ARG_DST, dst_memory } });

  // reorder or copy the output
  if (dst_memory != user_dst_memory) {
    net.push_back(reorder(dst_memory, user_dst_memory));
    net_args.push_back({ { DNNL_ARG_FROM, dst_memory },
        { DNNL_ARG_TO, user_dst_memory } });
  }

  // run
  assert(net.size() == net_args.size() && "something is missing");
  for (size_t i = 0; i < net.size(); ++i)
      net.at(i).execute(s, net_args.at(i));

  s.wait();

#ifdef DUMP_FLAG
  if (dump_idx == 0) {
    write_bianry_file(prefix + std::string("_out.bin"),
        (const char *)output, m * n * sizeof(float));
  }
  dump_idx ++;
#endif // DUMP_FLAG

  return 0;
}

} // namespace tpu_mlir
