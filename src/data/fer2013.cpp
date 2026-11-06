/**
 * fer2013.cpp
 * Author: Miguel Mochizuki Silva
 * Description: Dataset reading the FER-2013 facial-expression CSV (emotion,pixels,Usage)
 */

#include "nn/data/fer2013.hpp"

#include <algorithm>
#include <cstdlib>
#include <fstream>
#include <stdexcept>

namespace nn {

namespace {

[[noreturn]] void fail(size_t line, const std::string& reason) {
	throw std::runtime_error("FER-2013 CSV line " + std::to_string(line) + ": " + reason);
}

// Splits one CSV line into fields, honouring double-quoted fields and "" escapes
std::vector<std::string> split_csv(const std::string& line, size_t line_no) {
	std::vector<std::string> fields(1);
	bool quoted = false;
	for (size_t i = 0; i < line.size(); i++) {
		char ch = line[i];
		if (quoted) {
			if (ch == '"') {
				if (i + 1 < line.size() && line[i + 1] == '"') {
					fields.back() += '"';
					i++;
				} else {
					quoted = false;
				}
			} else {
				fields.back() += ch;
			}
		} else if (ch == '"') {
			quoted = true;
		} else if (ch == ',') {
			fields.emplace_back();
		} else {
			fields.back() += ch;
		}
	}
	if (quoted) {
		fail(line_no, "unterminated quoted field");
	}
	return fields;
}

// Parses a non-negative integer that must span the whole string
bool parse_uint(const std::string& s, long& out) {
	if (s.empty() || s.size() > 9) return false;
	long v = 0;
	for (char ch : s) {
		if (ch < '0' || ch > '9') return false;
		v = v * 10 + (ch - '0');
	}
	out = v;
	return true;
}

}  // namespace

Fer2013Dataset::Fer2013Dataset(const std::string& csv_path, const std::string& usage, size_t image_size)
	: image_size_(image_size) {
	if (image_size == 0) {
		throw std::runtime_error("FER-2013: image_size must be > 0");
	}
	std::ifstream in(csv_path, std::ios::binary);
	if (!in) {
		throw std::runtime_error("FER-2013: cannot open " + csv_path);
	}

	size_t pixel_count = image_size * image_size;
	size_t col_emotion = 0, col_pixels = 0, col_usage = 0;
	bool have_header = false;
	std::string line;
	size_t line_no = 0;
	while (std::getline(in, line)) {
		line_no++;
		if (!line.empty() && line.back() == '\r') {
			line.pop_back();
		}
		if (line.empty()) {
			continue;
		}
		std::vector<std::string> fields = split_csv(line, line_no);

		if (!have_header) {
			bool e = false, p = false, u = false;
			for (size_t i = 0; i < fields.size(); i++) {
				if (fields[i] == "emotion") { col_emotion = i; e = true; }
				else if (fields[i] == "pixels") { col_pixels = i; p = true; }
				else if (fields[i] == "Usage") { col_usage = i; u = true; }
			}
			if (!e || !p) {
				fail(line_no, "header must contain the columns emotion and pixels");
			}
			if (!u && !usage.empty()) {
				fail(line_no, "header has no Usage column but a usage filter was given");
			}
			have_header = true;
			continue;
		}

		size_t needed = std::max(col_emotion, col_pixels) + 1;
		if (!usage.empty()) {
			needed = std::max(needed, col_usage + 1);
		}
		if (fields.size() < needed) {
			fail(line_no, "expected at least " + std::to_string(needed) + " fields, found " + std::to_string(fields.size()));
		}
		if (!usage.empty() && fields[col_usage] != usage) {
			continue;
		}

		long emotion = 0;
		if (!parse_uint(fields[col_emotion], emotion) || emotion > 6) {
			fail(line_no, "emotion must be an integer in [0, 6], got '" + fields[col_emotion] + "'");
		}

		size_t before = pixels_.size();
		const std::string& px = fields[col_pixels];
		size_t pos = 0;
		while (pos < px.size()) {
			while (pos < px.size() && px[pos] == ' ') pos++;
			if (pos >= px.size()) break;
			size_t end = px.find(' ', pos);
			if (end == std::string::npos) end = px.size();
			long v = 0;
			if (!parse_uint(px.substr(pos, end - pos), v) || v > 255) {
				fail(line_no, "pixel values must be integers in [0, 255], got '" + px.substr(pos, end - pos) + "'");
			}
			pixels_.push_back(static_cast<float>(v) / 255.0f);
			pos = end;
		}
		size_t got = pixels_.size() - before;
		if (got != pixel_count) {
			pixels_.resize(before);
			fail(line_no, "expected " + std::to_string(pixel_count) + " pixels, found " + std::to_string(got));
		}
		labels_.push_back(static_cast<float>(emotion));
	}
	if (!have_header) {
		throw std::runtime_error("FER-2013: " + csv_path + " is empty");
	}
}

std::pair<Tensor, Tensor> Fer2013Dataset::get(size_t index) const {
	if (index >= labels_.size()) {
		throw TensorIndexError();
	}
	size_t n = image_size_ * image_size_;
	Tensor image({1, image_size_, image_size_});
	std::copy(pixels_.begin() + index * n, pixels_.begin() + (index + 1) * n, image.data().begin());
	return {image, Tensor({}, labels_[index])};
}

std::vector<size_t> Fer2013Dataset::class_counts() const {
	std::vector<size_t> counts(7, 0);
	for (float l : labels_) {
		counts[static_cast<size_t>(l)]++;
	}
	return counts;
}

std::vector<size_t> Fer2013Dataset::labels() const {
	std::vector<size_t> out(labels_.size());
	for (size_t i = 0; i < out.size(); i++) out[i] = static_cast<size_t>(labels_[i]);
	return out;
}

}  // namespace nn
