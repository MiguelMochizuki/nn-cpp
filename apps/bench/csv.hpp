/**
 * csv.hpp
 * Author: Miguel Mochizuki Silva
 * Description: Minimal CSV writer for benchmark results
 */

#pragma once

#include <fstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace bench {

/**
 * Writes comma-separated rows to a file; every row must match the header width.
 * Cells are written verbatim (no quoting).
 *
 * Attributes:
 * std::ofstream out_: Output stream
 * size_t width_: Column count fixed by header()
 */
class CsvWriter {
public:
	/**
	 * Opens the output file for writing.
	 *
	 * Parameters:
	 * const std::string& path: Destination file; throws std::runtime_error if it cannot be opened
	 */
	explicit CsvWriter(const std::string& path) : out_(path) {
		if (!out_) throw std::runtime_error("CsvWriter: cannot open " + path);
	}

	/**
	 * Writes the header line and fixes the expected column count.
	 *
	 * Parameters:
	 * const std::vector<std::string>& cols: Column names
	 */
	void header(const std::vector<std::string>& cols) {
		width_ = cols.size();
		write(cols);
	}

	/**
	 * Writes one data row.
	 *
	 * Parameters:
	 * const std::vector<std::string>& cells: Cell values; throws std::invalid_argument if the count differs from the header
	 */
	void row(const std::vector<std::string>& cells) {
		if (cells.size() != width_) throw std::invalid_argument("CsvWriter: wrong cell count");
		write(cells);
	}

private:
	/**
	 * Writes a comma-joined line.
	 *
	 * Parameters:
	 * const std::vector<std::string>& cells: Values to join
	 */
	void write(const std::vector<std::string>& cells) {
		for (size_t i = 0; i < cells.size(); ++i) out_ << (i ? "," : "") << cells[i];
		out_ << '\n';
	}

	std::ofstream out_;
	size_t width_ = 0;
};

}  // namespace bench
