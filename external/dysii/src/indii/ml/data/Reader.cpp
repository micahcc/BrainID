#include "Reader.hpp"

#include <fstream>

using namespace indii::ml::data;

Reader::Reader(std::istream* in) : in(in), ownStream(false) {
  //
}

Reader::Reader(const std::string& file) : ownStream(true) {
  in = new std::ifstream(file.c_str());
}

Reader::Reader(std::istream* in, unsigned int col) : in(in),
    ownStream(false) {
  cols.push_back(col);
}

Reader::Reader(const std::string& file, unsigned int col) : ownStream(true) {
  in = new std::ifstream(file.c_str());
  cols.push_back(col);
}

Reader::Reader(std::istream* in, const std::vector<unsigned int>& cols) :
    in(in), cols(cols), ownStream(false) {
  //
}

Reader::Reader(const std::string& file, const std::vector<unsigned int>& cols)
    : cols(cols), ownStream(true) {
  in = new std::ifstream(file.c_str());
}

Reader::~Reader() {
  if (ownStream) {
    static_cast<std::ifstream*>(in)->close();
    delete in;
  }
}

