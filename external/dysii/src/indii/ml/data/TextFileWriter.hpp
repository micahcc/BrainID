#ifndef INDII_ML_DATA_TEXTFILEWRITER_HPP
#define INDII_ML_DATA_TEXTFILEWRITER_HPP

#include "Writer.hpp"

namespace indii {
  namespace ml {
    namespace data {
    
/**
 * Writer for white-space delimited text files.
 *
 * @author Lawrence Murray <lawrence@indii.org>
 * @version $Rev: 516 $
 * @date $Date: 2008-08-12 18:08:30 +0100 (Tue, 12 Aug 2008) $
 */
class TextFileWriter : public Writer {
public:
  /**
   * Construct from output stream.
   *
   * @param out Stream to which to write.
   */
  TextFileWriter(std::ostream* out);

  /**
   * Construct from file
   *
   * @param file Name of file to which to write.
   */
  TextFileWriter(const std::string file);

  /**
   * Destructor.
   */
  virtual ~TextFileWriter();

  virtual void write(const double value);

  virtual void write(const indii::ml::aux::vector& values);

  virtual void write(const indii::ml::aux::matrix& values);

  virtual void write(const indii::ml::aux::symmetric_matrix& values);

  virtual void writeLine();
  
  using Writer::writeLine;

};

    }
  }
}

#endif

