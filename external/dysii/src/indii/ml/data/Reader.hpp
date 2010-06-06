#ifndef INDII_ML_DATA_READER_HPP
#define INDII_ML_DATA_READER_HPP

#include "../aux/vector.hpp"
#include "../aux/matrix.hpp"

#include <iostream>
#include <string>
#include <vector>

namespace indii {
  namespace ml {
    namespace data {

/**
 * Abstract reader for %data files.
 *
 * @author Lawrence Murray <lawrence@indii.org>
 * @version $Rev: 582 $
 * @date $Date: 2008-12-15 17:03:50 +0000 (Mon, 15 Dec 2008) $
 */
class Reader {
public:
  /**
   * Construct new reader from an input stream, where all columns are of
   * interest.
   *
   * @param in Stream from which to read.
   */
  Reader(std::istream* in);

  /**
   * Construct new reader from a file, where all columns are of interest.
   *
   * @param file Name of file from which to read.
   */
  Reader(const std::string& file);

  /**
   * Construct new reader from an input stream, where only a
   * single column is of interest.
   *
   * @param in Stream from which to read.
   * @param col Index of the column of interest.
   */
  Reader(std::istream* in, unsigned int col);

  /**
   * Construct new reader from a file, where only a single column
   * is of interest.
   *
   * @param file Name of file from which to read.
   * @param col Index of the column of interest .
   */
  Reader(const std::string& file, unsigned int col);

  /**
   * Construct new reader from an input stream, where only a
   * subset of columns are of interest.
   *
   * @param in Stream from which to read.
   * @param cols Indices of the columns of interest from the input
   * stream, in the order of interest.
   */
  Reader(std::istream* in, const std::vector<unsigned int>& cols);

  /**
   * Construct new reader from a file, where only a subset of
   * columns are of interest.
   *
   * @param file Name of file from which to read.
   * @param cols Indices of the columns of interest from the input
   * stream, in the order of interest.
   */
  Reader(const std::string& file, const std::vector<unsigned int>& cols);

  /**
   * Destructor. The input stream is closed if the object was created
   * using a file name, but left open otherwise.
   */
  virtual ~Reader();

  /**
   * Read next value.
   *
   * @param into Double into which to read the value.
   *
   * @return Number of values actually read. Will be 1 if a value is
   * successfully read, and 0 if the end of the stream is reached
   * during reading.
   *
   * One value is read from the input stream into @c into. If a
   * particular column or subset of columns of interest have been
   * specified, all others are skipped during the reading.
   */
  virtual unsigned int read(double* into) = 0;

  /**
   * Read next values into vector.
   *
   * @param into Vector into which to read the values.
   *
   * @return Number of values actually read. Will be less than the
   * size of the given vector if the end of the stream is reached
   * during reading.
   *
   * <tt>into.size()</tt> values are read from the input stream into
   * @c into. If a particular column or subset of columns of interest
   * have been specified, all others are skipped during the reading.
   */
  virtual unsigned int read(indii::ml::aux::vector* into) = 0;

  /**
   * Read next values into matrix.
   *
   * @param into Matrix into which to read the values.
   *
   * @return Number of values actually read. Will be less than the
   * size of the given matrix if the end of the stream is reached
   * during reading.
   *
   * <tt>into.size1() * into.size2()</tt> values are read from the
   * input stream into @c into in column-wise fashion. If a particular
   * column or subset of columns of interest have been specified, all
   * others are skipped during the reading.
   */
  virtual unsigned int read(indii::ml::aux::matrix* into) = 0;

  /**
   * Read next values into symmetric matrix.
   *
   * @param into Matrix into which to read the values.
   *
   * @return Number of values actually read. Will be less than the
   * size of the lower triangular portion of the given matrix if the
   * end of the stream is reached during reading.
   *
   * <tt>0.5 * (into.size1() * into.size1() + into.size1())</tt>
   * values are read from the input stream into the lower triangle of
   * @c into in column-wise fashion. If a particular column or subset of
   * columns of interest have been specified, all others are skipped
   * during the reading.
   */
  virtual unsigned int read(indii::ml::aux::symmetric_matrix* into) = 0;

protected:
  /**
   * The input stream.
   */
  std::istream* in;

  /**
   * Columns of interest. Empty if all columns are of interest.
   */
  std::vector<unsigned int> cols;

private:
  /**
   * Does object own the input stream?
   */
  const bool ownStream;

};

    }
  }
}

#endif
