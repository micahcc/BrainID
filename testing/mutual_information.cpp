#include "itkOrientedImage.h"
#include "itkImageFileWriter.h"
#include "itkImageFileReader.h"
#include "tools.h"

template <typename T>
void writeImage(std::string name, typename T::Pointer in)
{
    std::cout << "Writing " << name << std::endl;
    typename itk::ImageFileWriter<T>::Pointer writer;
    writer = itk::ImageFileWriter<T>::New();
    writer->SetFileName( name );
    writer->SetInput( in );
    std::cout << "Done " << std::endl;
    writer->Update();
}

template <typename T>
typename T::Pointer readImage(std::string name)
{
    std::cout << "Reading " << name << std::endl;
    typename itk::ImageFileReader<T>::Pointer reader;
    reader = itk::ImageFileReader<T>::New();
    reader->SetFileName( name );
    reader->Update();
    std::cout << "Done " << std::endl;
    return reader->GetOutput();
}

int main(int argc, char* argv[])
{
    if(argc != 4) {
        std::cerr << "Usage:" << std::endl << argv[0] << " <img1> <img2> <out>" 
                    << std::endl; 
        return -2;
    }
    Image4DType::Pointer img1 = readImage<Image4DType>(argv[1]);
    Image4DType::Pointer img2 = readImage<Image4DType>(argv[2]);
    try{ 
        Image4DType::Pointer tmp = readImage<Image4DType>(argv[3]);
        std::cerr << "File " << argv[3] << " exists!" << std::endl;
        std::cerr << "Usage:" << std::endl << argv[0] << " <img1> <img2> <out>" 
                    << std::endl; 
        return -1;
    } catch (...) { }
        
    Image3DType::Pointer out = mutual_info(6, 6, img1, img2);
    writeImage<Image3DType>(argv[3], out);
}
