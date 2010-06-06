#include "TextFileReader.hpp"

namespace aux = indii::ml::aux;

using namespace indii::ml::data;

TextFileReader::TextFileReader(std::istream* in) : Reader(in) {
  nextLine();
}

TextFileReader::TextFileReader(const std::string& file) : Reader(file) {
  nextLine();
}

TextFileReader::TextFileReader(std::istream* in, unsigned int col) :
    Reader(in, col) {
  nextLine();
}

TextFileReader::TextFileReader(const std::string& file, unsigned int col) :
    Reader(file, col) {
  nextLine();
}

TextFileReader::TextFileReader(std::istream* in,
    const std::vector<unsigned int>& cols) : Reader(in, cols) {
  nextLine();
}

TextFileReader::TextFileReader(const std::string& file,
    const std::vector<unsigned int>& cols) : Reader(file, cols) {
  nextLine();
}

TextFileReader::~TextFileReader() {
  //
}

unsigned int TextFileReader::read(double* into) {
  if (in->eof()) {
    return 0;
  } else {
    if (cols.empty()) {
      /* all values of interest, return next */
      *into = values[nextIndex];
      nextIndex++;
      if (nextIndex >= values.size()) {
        nextLine();
      }
    } else {
      /* return next value of interest */
      *into = values[cols[nextIndex]];
      nextIndex++;
      if (nextIndex >= cols.size() || cols[nextIndex] > values.size()) {
        nextLine();
      }
    }
    return 1;
  }
}

unsigned int TextFileReader::read(aux::vector* into) {
  double x;
  unsigned int i = 0;
  unsigned int size = into->size();

  while (i < size && read(&x) > 0) {
    (*into)(i) = x;
    i++;
  }
  return i;
}

unsigned int TextFileReader::read(aux::matrix* into) {
  double x;
  unsigned int numRead = 0;
  unsigned int i = 0, j = 0;
  unsigned int size1 = into->size1(), size2 = into->size2();

  for (j = 0; j < size2; j++) {
    for (i = 0; i < size1 && read(&x) > 0; i++) {
      numRead++;
      (*into)(i,j) = x;
    }
  }

  return numRead;
}

unsigned int TextFileReader::read(aux::symmetric_matrix* into) {
  double x;
  unsigned int numRead = 0;
  unsigned int i, j;
  unsigned int size = into->size1();
  for (j = 0; j < size; j++) {
    for (i = j; i < size && read(&x) > 0; i++) {
      numRead++;
      (*into)(i,j) = x;
    }
  }

  return numRead;
}

void TextFileReader::nextLine() {
  double value;
  std::string line;
  std::stringstream lineStream;

  nextIndex = 0;
  values.clear();

  std::getline(*in, line);
  if (!in->eof()) {
    /* read in all values on this line */
    lineStream.str(line);
    do {
      lineStream >> value;
      values.push_back(value);
    } while (!lineStream.eof());
  }
}

