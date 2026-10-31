/**
 * checkpoint.cpp
 * Author: Miguel Mochizuki Silva
 * Description: Portable binary checkpoints of a Module's parameters and an Optimizer's state
 */

#include "nn/checkpoint.hpp"

#include <algorithm>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iterator>
#include <limits>
#include <set>
#include <stdexcept>

static_assert(std::numeric_limits<float>::is_iec559 && sizeof(float) == 4, "checkpoints require IEEE-754 float32");

namespace nn {

namespace {

const char MAGIC[4] = {'N', 'N', 'C', 'K'};
const uint32_t VERSION = 1;
const uint32_t MAX_NAME = 4096;
const uint32_t MAX_NDIM = 8;

struct Entry {
	std::string name;
	bool is_int = false;
	int64_t ivalue = 0;
	std::vector<size_t> shape;
	std::vector<float> data;
};

/**
 * Appends an unsigned integer in little-endian byte order
 *
 * Parameters:
 * std::vector<uint8_t>& out: Buffer to append to
 * uint64_t v: Value to write
 * int bytes: Number of bytes to emit
 */
void put_uint(std::vector<uint8_t>& out, uint64_t v, int bytes) {
	for (int i = 0; i < bytes; i++) out.push_back(static_cast<uint8_t>((v >> (8 * i)) & 0xFF));
}

/**
 * Serializes checkpoint entries to the file format (magic, version, count, then entries)
 *
 * Parameters:
 * const std::vector<Entry>& entries: Named integers and tensors to store
 *
 * Returns std::vector<uint8_t>: The encoded bytes
 */
std::vector<uint8_t> encode(const std::vector<Entry>& entries) {
	std::vector<uint8_t> out(MAGIC, MAGIC + 4);
	put_uint(out, VERSION, 4);
	put_uint(out, entries.size(), 4);
	for (const Entry& e : entries) {
		put_uint(out, e.name.size(), 4);
		out.insert(out.end(), e.name.begin(), e.name.end());
		out.push_back(e.is_int ? 1 : 0);
		if (e.is_int) {
			put_uint(out, static_cast<uint64_t>(e.ivalue), 8);
			continue;
		}
		put_uint(out, e.shape.size(), 4);
		for (size_t d : e.shape) put_uint(out, d, 8);
		for (float f : e.data) {
			uint32_t bits;
			std::memcpy(&bits, &f, 4);
			put_uint(out, bits, 4);
		}
	}
	return out;
}

// Bounds-checked cursor over the file bytes; every read throws on truncation.
class Reader {
public:
	explicit Reader(const std::vector<uint8_t>& b) : b_(b) {}

	size_t remaining() const { return b_.size() - pos_; }

	uint64_t uint(int bytes) {
		need(bytes);
		uint64_t v = 0;
		for (int i = 0; i < bytes; i++) v |= static_cast<uint64_t>(b_[pos_ + i]) << (8 * i);
		pos_ += bytes;
		return v;
	}

	std::string str(size_t n) {
		need(n);
		std::string s(b_.begin() + pos_, b_.begin() + pos_ + n);
		pos_ += n;
		return s;
	}

private:
	void need(size_t n) const {
		if (n > remaining()) throw std::runtime_error("load_checkpoint: unexpected end of file");
	}

	const std::vector<uint8_t>& b_;
	size_t pos_ = 0;
};

/**
 * Parses checkpoint bytes back into entries
 *
 * Parameters:
 * const std::vector<uint8_t>& bytes: Whole file contents
 *
 * Returns std::vector<Entry>: Decoded entries; throws std::runtime_error on a bad magic number,
 * unsupported version, duplicate name or truncated data
 */
std::vector<Entry> decode(const std::vector<uint8_t>& bytes) {
	Reader r(bytes);
	if (r.str(4) != std::string(MAGIC, 4)) throw std::runtime_error("load_checkpoint: not a checkpoint file");
	if (r.uint(4) != VERSION) throw std::runtime_error("load_checkpoint: unsupported checkpoint version");
	uint64_t count = r.uint(4);
	std::vector<Entry> entries;
	std::set<std::string> seen;
	for (uint64_t i = 0; i < count; i++) {
		Entry e;
		uint64_t name_len = r.uint(4);
		if (name_len > MAX_NAME || name_len > r.remaining()) throw std::runtime_error("load_checkpoint: bad entry name length");
		e.name = r.str(name_len);
		if (!seen.insert(e.name).second) throw std::runtime_error("load_checkpoint: duplicate entry " + e.name);
		uint64_t kind = r.uint(1);
		if (kind == 1) {
			e.is_int = true;
			e.ivalue = static_cast<int64_t>(r.uint(8));
		} else if (kind == 0) {
			uint64_t ndim = r.uint(4);
			if (ndim > MAX_NDIM) throw std::runtime_error("load_checkpoint: bad rank for " + e.name);
			uint64_t elems = 1;
			for (uint64_t d = 0; d < ndim; d++) {
				uint64_t dim = r.uint(8);
				// Each element needs 4 file bytes, so the file size bounds any honest element count.
				if (dim != 0 && elems > r.remaining() / 4 / dim) throw std::runtime_error("load_checkpoint: bad shape for " + e.name);
				elems *= dim;
				e.shape.push_back(dim);
			}
			if (elems > r.remaining() / 4) throw std::runtime_error("load_checkpoint: unexpected end of file");
			e.data.resize(elems);
			for (float& f : e.data) {
				uint32_t bits = static_cast<uint32_t>(r.uint(4));
				std::memcpy(&f, &bits, 4);
			}
		} else {
			throw std::runtime_error("load_checkpoint: bad entry kind for " + e.name);
		}
		entries.push_back(std::move(e));
	}
	if (r.remaining() != 0) throw std::runtime_error("load_checkpoint: trailing bytes after last entry");
	return entries;
}

/**
 * Wraps a tensor as a named checkpoint entry
 *
 * Parameters:
 * const std::string& name: Entry name
 * const Tensor& t: Tensor to store
 *
 * Returns Entry: Entry holding the tensor's shape and data
 */
Entry tensor_entry(const std::string& name, const Tensor& t) {
	Entry e;
	e.name = name;
	e.shape = t.shape();
	e.data = t.data();
	return e;
}

}  // namespace

void save_checkpoint(const std::string& path, const Module& model, const Optimizer* optimizer, int64_t step) {
	std::vector<Entry> entries;
	for (const auto& p : model.named_parameters()) entries.push_back(tensor_entry("model." + p.first, p.second.data()));
	if (optimizer) {
		for (const auto& st : optimizer->state()) entries.push_back(tensor_entry("optim." + st.first, st.second));
	}
	Entry s;
	s.name = "step";
	s.is_int = true;
	s.ivalue = step;
	entries.push_back(s);
	std::vector<uint8_t> bytes = encode(entries);

	const std::string tmp = path + ".tmp";
	{
		std::ofstream out(tmp, std::ios::binary | std::ios::trunc);
		out.write(reinterpret_cast<const char*>(bytes.data()), bytes.size());
		out.flush();
		if (!out) {
			out.close();
			std::remove(tmp.c_str());
			throw std::runtime_error("save_checkpoint: cannot write " + tmp);
		}
	}
	std::error_code ec;
	std::filesystem::rename(tmp, path, ec);
	if (ec) {
		std::remove(tmp.c_str());
		throw std::runtime_error("save_checkpoint: cannot replace " + path + ": " + ec.message());
	}
}

int64_t load_checkpoint(const std::string& path, Module& model, Optimizer* optimizer) {
	std::ifstream in(path, std::ios::binary);
	if (!in) throw std::runtime_error("load_checkpoint: cannot open " + path);
	std::vector<uint8_t> bytes((std::istreambuf_iterator<char>(in)), std::istreambuf_iterator<char>());
	std::vector<Entry> entries = decode(bytes);

	auto params = model.named_parameters();
	std::string problems;
	std::set<std::string> expected;
	for (const auto& p : params) expected.insert("model." + p.first);
	int64_t step = 0;
	bool has_step = false;
	OptimizerState optimizer_state;
	for (const Entry& e : entries) {
		if (e.name == "step" && e.is_int) {
			step = e.ivalue;
			has_step = true;
		} else if (e.name.rfind("optim.", 0) == 0) {
			if (e.is_int) problems += " bad-kind:" + e.name;
			else optimizer_state.emplace_back(e.name.substr(6), Tensor(e.shape, e.data));
		} else if (!expected.count(e.name)) {
			problems += " unexpected:" + e.name;
		}
	}
	if (!has_step) problems += " missing:step";
	for (const auto& p : params) {
		std::string name = "model." + p.first;
		auto it = std::find_if(entries.begin(), entries.end(), [&](const Entry& e) { return e.name == name; });
		if (it == entries.end() || it->is_int) problems += " missing:" + name;
		else if (it->shape != p.second.data().shape()) problems += " shape-mismatch:" + name;
	}
	if (!problems.empty()) throw std::runtime_error("load_checkpoint: checkpoint does not match model:" + problems);

	// The optimizer validates all-or-nothing and throws before any model weight is touched.
	if (optimizer) {
		if (optimizer_state.empty()) throw std::runtime_error("load_checkpoint: checkpoint has no optimizer state");
		optimizer->load_state(optimizer_state);
	}
	for (const auto& p : params) {
		std::string name = "model." + p.first;
		auto it = std::find_if(entries.begin(), entries.end(), [&](const Entry& e) { return e.name == name; });
		p.second.data().data() = it->data;
	}
	return step;
}

}  // namespace nn
