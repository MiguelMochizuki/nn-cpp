/**
 * vmath.cpp
 * Author: Miguel Mochizuki Silva
 * Description: Elementwise single-precision math kernels (tanh, exp) with one result per input value
 */

#include "nn/kernels/vmath.hpp"

#include <cmath>
#include <cstdint>
#include <cstring>

#if defined(__AVX2__)
#define NN_VMATH_AVX2 1
#include <immintrin.h>
#endif

namespace nn {
namespace kernels {

namespace {

// Cephes expf: x = n ln2 + r with |r| <= ln2 / 2, e^r = 1 + r + r^2 P(r), ln2 split in two for an exact n ln2
// Below kExpLo the result is under FLT_MIN and returned as 0; above kExpHi it is returned as infinity
constexpr float kExpLo = -87.3f, kExpHi = 88.3f;
constexpr float kLog2e = 1.44269504088896341f;
constexpr float kLn2Hi = 0.693359375f, kLn2Lo = -2.12194440e-4f;
constexpr float kP0 = 1.9875691500e-4f, kP1 = 1.3981999507e-3f, kP2 = 8.3334519073e-3f;
constexpr float kP3 = 4.1665795894e-2f, kP4 = 1.6666665459e-1f, kP5 = 5.0000001201e-1f;
constexpr float kTanhMax = 9.0f;

#ifdef NN_VMATH_AVX2
/**
 * exp_one on 8 lanes: the same operations in the same order, so each lane has exp_one's bits
 *
 * Parameters:
 * __m256 x: Arguments
 *
 * Returns __m256: e^x per lane
 */
inline __m256 exp_lanes(__m256 x) {
	// max(lo, x) and min(hi, x) return x when it is NaN, as the scalar comparisons do
	__m256 z = _mm256_min_ps(_mm256_set1_ps(kExpHi), _mm256_max_ps(_mm256_set1_ps(kExpLo), x));
	__m256 n = _mm256_round_ps(_mm256_mul_ps(z, _mm256_set1_ps(kLog2e)), _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC);
	__m256 r = _mm256_sub_ps(z, _mm256_mul_ps(n, _mm256_set1_ps(kLn2Hi)));
	r = _mm256_sub_ps(r, _mm256_mul_ps(n, _mm256_set1_ps(kLn2Lo)));
	__m256 p = _mm256_set1_ps(kP0);
	p = _mm256_add_ps(_mm256_mul_ps(p, r), _mm256_set1_ps(kP1));
	p = _mm256_add_ps(_mm256_mul_ps(p, r), _mm256_set1_ps(kP2));
	p = _mm256_add_ps(_mm256_mul_ps(p, r), _mm256_set1_ps(kP3));
	p = _mm256_add_ps(_mm256_mul_ps(p, r), _mm256_set1_ps(kP4));
	p = _mm256_add_ps(_mm256_mul_ps(p, r), _mm256_set1_ps(kP5));
	__m256 e = _mm256_add_ps(_mm256_add_ps(_mm256_mul_ps(p, _mm256_mul_ps(r, r)), r), _mm256_set1_ps(1.0f));
	__m256i pow2 = _mm256_slli_epi32(_mm256_add_epi32(_mm256_cvtps_epi32(n), _mm256_set1_epi32(127)), 23);
	__m256 y = _mm256_mul_ps(e, _mm256_castsi256_ps(pow2));
	y = _mm256_blendv_ps(y, _mm256_setzero_ps(), _mm256_cmp_ps(x, _mm256_set1_ps(kExpLo), _CMP_LT_OQ));
	y = _mm256_blendv_ps(y, _mm256_set1_ps(HUGE_VALF), _mm256_cmp_ps(x, _mm256_set1_ps(kExpHi), _CMP_GT_OQ));
	return _mm256_blendv_ps(y, x, _mm256_cmp_ps(x, x, _CMP_UNORD_Q));
}

/**
 * tanh_one on 8 lanes, with tanh_one's bits per lane
 *
 * Parameters:
 * __m256 x: Arguments
 *
 * Returns __m256: tanh(x) per lane
 */
inline __m256 tanh_lanes(__m256 x) {
	__m256 z = _mm256_min_ps(_mm256_set1_ps(kTanhMax), _mm256_max_ps(_mm256_set1_ps(-kTanhMax), x));
	__m256 e = exp_lanes(_mm256_add_ps(z, z));
	const __m256 one = _mm256_set1_ps(1.0f);
	return _mm256_div_ps(_mm256_sub_ps(e, one), _mm256_add_ps(e, one));
}
#endif

}  // namespace

float exp_one(float x) {
	if (x != x) return x;
	if (x < kExpLo) return 0.0f;
	if (x > kExpHi) return HUGE_VALF;
	float z = x;
	// nearbyint rounds half to even in the default rounding mode, like _MM_FROUND_TO_NEAREST_INT
	float n = std::nearbyint(z * kLog2e);
	float r = z - n * kLn2Hi;
	r = r - n * kLn2Lo;
	float p = kP0;
	p = p * r + kP1;
	p = p * r + kP2;
	p = p * r + kP3;
	p = p * r + kP4;
	p = p * r + kP5;
	float e = (p * (r * r) + r) + 1.0f;
	uint32_t bits = static_cast<uint32_t>(static_cast<int32_t>(n) + 127) << 23;
	float pow2;
	std::memcpy(&pow2, &bits, sizeof pow2);
	return e * pow2;
}

float tanh_one(float x) {
	if (x != x) return x;
	float z = x < -kTanhMax ? -kTanhMax : x;
	z = kTanhMax < z ? kTanhMax : z;
	float e = exp_one(z + z);
	return (e - 1.0f) / (e + 1.0f);
}

void exp_inplace(float* x, size_t n) {
	size_t i = 0;
#ifdef NN_VMATH_AVX2
	for (; i + 8 <= n; i += 8) _mm256_storeu_ps(x + i, exp_lanes(_mm256_loadu_ps(x + i)));
#endif
	for (; i < n; i++) x[i] = exp_one(x[i]);
}

void tanh_inplace(float* x, size_t n) {
	size_t i = 0;
#ifdef NN_VMATH_AVX2
	for (; i + 8 <= n; i += 8) _mm256_storeu_ps(x + i, tanh_lanes(_mm256_loadu_ps(x + i)));
#endif
	for (; i < n; i++) x[i] = tanh_one(x[i]);
}

}  // namespace kernels
}  // namespace nn
