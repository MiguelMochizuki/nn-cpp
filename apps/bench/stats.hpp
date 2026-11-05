/**
 * stats.hpp
 * Author: Miguel Mochizuki Silva
 * Description: Robust summary statistics (median, relative spread) for benchmark timings
 */

#pragma once

#include <algorithm>
#include <stdexcept>
#include <vector>

namespace bench {

/**
 * Linear-interpolated quantile of an already sorted, non-empty vector.
 *
 * @param s sorted samples
 * @param q quantile in [0, 1]
 * @return the interpolated value
 */
inline double quantile_sorted(const std::vector<double>& s, double q) {
	const double pos = q * static_cast<double>(s.size() - 1);
	const size_t lo = static_cast<size_t>(pos);
	const size_t hi = std::min(lo + 1, s.size() - 1);
	return s[lo] + (pos - static_cast<double>(lo)) * (s[hi] - s[lo]);
}

/**
 * Median of a sample set.
 *
 * @param v samples (taken by value, sorted internally)
 * @return the median
 * @throws std::invalid_argument if v is empty
 */
inline double median(std::vector<double> v) {
	if (v.empty()) throw std::invalid_argument("median: empty input");
	std::sort(v.begin(), v.end());
	return quantile_sorted(v, 0.5);
}

/**
 * Relative spread: interquartile range divided by the median.
 *
 * @param v samples (taken by value, sorted internally)
 * @return IQR / median, or 0.0 when there are fewer than 2 samples
 */
inline double spread(std::vector<double> v) {
	if (v.size() < 2) return 0.0;
	std::sort(v.begin(), v.end());
	return (quantile_sorted(v, 0.75) - quantile_sorted(v, 0.25)) / quantile_sorted(v, 0.5);
}

}  // namespace bench
