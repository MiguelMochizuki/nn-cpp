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
 */
class CsvWriter {
public:
	/**
	 * Opens the output file for writing.
	 *
	 * @param path destination file
	 * @throws std::runtime_error if the file cannot be opened
	 */
	explicit CsvWriter(const std::string& path) : out_(path) {
		if (!out_) throw std::runtime_error("CsvWriter: cannot open " + path);
	}

	/**
	 * Writes the header line and fixes the expected column count.
	 *
	 * @param cols column names
	 */
	void header(const std::vector<std::string>& cols) {
		width_ = cols.size();
		write(cols);
	}

	/**
	 * Writes one data row.
	 *
	 * @param cells cell values
	 * @throws std::invalid_argument if the cell count differs from the header
	 */
	void row(const std::vector<std::string>& cells) {
		if (cells.size() != width_) throw std::invalid_argument("CsvWriter: wrong cell count");
		write(cells);
	}

private:
	/**
	 * Writes a comma-joined line.
	 *
	 * @param cells values to join
	 */
	void write(const std::vector<std::string>& cells) {
		for (size_t i = 0; i < cells.size(); ++i) out_ << (i ? "," : "") << cells[i];
		out_ << '\n';
	}

	std::ofstream out_;
	size_t width_ = 0;
};

}  // namespace bench
