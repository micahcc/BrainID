#include "TextFileWriter.hpp"

using namespace indii::ml::data;

namespace aux = indii::ml::aux;

TextFileWriter::TextFileWriter(std::ostream* out) : Writer(out) {
  //
}

TextFileWriter::TextFileWriter(const std::string file) : Writer(file) {
  //
}

TextFileWriter::~TextFileWriter() {
  //
}

void TextFileWriter::write(const double value) {
  *out << '\t' << value;
}

void TextFileWriter::write(const aux::vector& values) {
  unsigned int i;
  unsigned int size = values.size();

  for (i = 0; i < size; i++) {
    write(values(i));
  }
}

void TextFileWriter::write(const aux::matrix& values) {
  unsigned int i, j;
  unsigned int size1 = values.size1(), size2 = values.size2();

  for (j = 0; j < size2; j++) {
    for (i = 0; i < size1; i++) {
      write(values(i,j));
    }
  }
}

void TextFileWriter::write(const aux::symmetric_matrix& values) {
  unsigned int i, j;
  unsigned int size1 = values.size1(), size2 = values.size2();

  for (j = 0; j < size2; j++) {
    for (i = j; i < size1; i++) {
      write(values(i,j));
    }
  }
}

void TextFileWriter::writeLine() {
  *out << std::endl;
}

