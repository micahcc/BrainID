#ifndef INDII_ML_DATA_TEXTFILEReader_HPP
#define INDII_ML_DATA_TEXTFILEReader_HPP

#include "Reader.hpp"

#include "../aux/vector.hpp"
#include "../aux/matrix.hpp"

#include <sstream>

namespace indii {
  namespace ml {
    namespace data {

/**
 * Reader for white-space delimited text files.
 *
 * @author Lawrence Murray <lawrence@indii.org>
 * @version $Rev: 582 $
 * @date $Date: 2008-12-15 17:03:50 +0000 (Mon, 15 Dec 2008) $
 */
class TextFileReader : public Reader {
public:
  /**
   * Construct new reader from input stream, where all columns are
   * of interest.
   *
   * @param in Stream from which to read.
   */
  TextFileReader(std::istream* in);

  /**
   * Construct new reader from file, where all columns are of
   * interest.
   *
   * @param file Name of file from which to read.
   */
  TextFileReader(const std::string& file);

  /**
   * Construct new reader from input stream, where only a single
   * column is of interest.
   *
   * @param in Stream from which to read.
   * @param col Index of column of interest.
   */
  TextFileReader(std::istream* in, unsigned int col);

  /**
   * Construct new reader from file, where only a single column is
   * of interest.
   *
   * @param file Name of file from which to read.
   * @param col Index of column of interest.
   */
  TextFileReader(const std::string& file, unsigned int col);

  /**
   * Construct new reader from input stream, where only a subset
   * of columns are of interest.
   *
   * @param in Stream from which to read.
   * @param cols Set of indices giving columns of interest.
   */
  TextFileReader(std::istream* in, const std::vector<unsigned int>& cols);

  /**
   * Construct new reader from file, where only a subset of columns
   * are of interest.
   *
   * @param file Name of file from which to read.
   * @param cols Set of indices giving columns of interest.
   */
  TextFileReader(const std::string& file,
      const std::vector<unsigned int>& cols);

  /**
   * Destructor.
   */
  ~TextFileReader();

  virtual unsigned int read(double* into);

  virtual unsigned int read(indii::ml::aux::vector* into);

  virtual unsigned int read(indii::ml::aux::matrix* into);

  virtual unsigned int read(indii::ml::aux::symmetric_matrix* into);

private:
  /**
   * Values on the current line.
   */
  std::vector<double> values;

  /**
   * Current index in the columns of interest vector.
   */
  unsigned int nextIndex;

  /** 
   * Move to next line in the stream.
   */
  void nextLine();

};

    }
  }
}

#endif
