#ifndef INDII_ML_DATA_WRITER_HPP
#define INDII_ML_DATA_WRITER_HPP

#include "../aux/vector.hpp"
#include "../aux/matrix.hpp"

#include <iostream>

namespace indii {
  namespace ml {
    namespace data {
    
/**
 * Abstract writer for %data files.
 *
 * @author Lawrence Murray <lawrence@indii.org>
 * @version $Rev: 516 $
 * @date $Date: 2008-08-12 18:08:30 +0100 (Tue, 12 Aug 2008) $
 */
class Writer {
public:
  /**
   * Construct from output stream.
   *
   * @param out Stream to which to write.
   */
  Writer(std::ostream* out);

  /**
   * Construct from file.
   *
   * @param file Name of file to which to write.
   */
  Writer(const std::string file);

  /**
   * Destructor. The output stream is closed if the object was created
   * using a file name, but left open otherwise.
   */
  virtual ~Writer();

  /**
   * Write single value.
   *
   * @param value Value to write.
   */
  virtual void write(const double value) = 0;

  /**
   * Write vector of values. All values are written to the current
   * line.
   *
   * @param values Vector of values to write.
   */
  virtual void write(const indii::ml::aux::vector& values) = 0;

  /**
   * Write matrix of values. All values are written to the current
   * line in column-wise fashion.
   *
   * @param values Matrix of values to write.
   */
  virtual void write(const indii::ml::aux::matrix& values) = 0;

  /**
   * Write symmetric matrix of values. All values from the lower
   * triangle of the matrix are written to the current line in
   * column-wise fashion.
   *
   * @param values Matrix of values to write.
   */
  virtual void write(const indii::ml::aux::symmetric_matrix& values) = 0;

  /**
   * Write the end of the current line and begin a new line.
   */
  virtual void writeLine() = 0;

  /**
   * Convenience method equivalent to calling write(const double) then
   * writeLine().
   */
  void writeLine(const double value);

  /**
   * Convenience method equivalent to calling write(const
   * indii::ml::aux::vector) then writeLine().
   */
  void writeLine(const indii::ml::aux::vector& values);

  /**
   * Convenience method equivalent to calling write(const
   * indii::ml::aux::matrix) then writeLine().
   */
  void writeLine(const indii::ml::aux::matrix& values);

  /**
   * Convenience method equivalent to calling write(const
   * indii::ml::aux::symmetric_matrix) then writeLine().
   */
  void writeLine(const indii::ml::aux::symmetric_matrix& values);

protected:
  /**
   * The output stream.
   */
  std::ostream* out;

private:
  /**
   * Does object own the output stream?
   */
  const bool ownStream;

};

    }
  }
}

#endif

