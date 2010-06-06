#include "Writer.hpp"

#include <iostream>
#include <fstream>
#include <assert.h>

using namespace indii::ml::data;

namespace aux = indii::ml::aux;

Writer::Writer(std::ostream* out) : out(out), ownStream(false) {
  //
}

Writer::Writer(const std::string file) : ownStream(true) {
  out = new std::ofstream(file.c_str());
}

Writer::~Writer() {
  if (ownStream) {
    static_cast<std::ofstream*>(out)->close();
    delete out;
  }
}

void Writer::writeLine(const double value) {
  write(value);
  writeLine();
}

void Writer::writeLine(const aux::vector& values) {
  write(values);
  writeLine();
}

void Writer::writeLine(const aux::matrix& values) {
  write(values);
  writeLine();
}

void Writer::writeLine(const aux::symmetric_matrix& values) {
  write(values);
  writeLine();
}

