//image readers
#include "itkImageFileReader.h"
#include <itkImageFileWriter.h>
#include "modNiftiImageIO.h"
#include "tools.h"

#include <iostream>
#include <fstream>
#include <string>
#include <sstream>

template <typename T>
void writeImage(std::string base, std::string name, typename T::Pointer in)
{
    base.append(name);
    typename itk::ImageFileWriter<T>::Pointer writer;
    writer = itk::ImageFileWriter<T>::New();
    writer->SetFileName( base );
    writer->SetInput( in );
    writer->Update();
}

template <typename T>
typename T::Pointer readImage(std::string base, std::string name)
{
    base.append(name);
    typename itk::ImageFileReader<T>::Pointer reader;
    reader = itk::ImageFileReader<T>::New();
    reader->SetFileName( base );
    reader->Update();
    return reader->GetOutput();
}

//unsigned int min(unsigned int a, unsigned int b)
//{
//    return a < b ? a : b;
//}
//
//unsigned int max(unsigned int a, unsigned int b)
//{
//    return b < a ? a : b;
//}

//template <class SrcType, class DstType >
//void copyInformation(typename SrcType::Pointer src, typename DstType::Pointer dst)
//{
//    typename SrcType::PointType srcOrigin = src->GetOrigin();
//    typename DstType::PointType dstOrigin = dst->GetOrigin();
//    
//    typename SrcType::DirectionType srcDir = src->GetDirection();
//    typename DstType::DirectionType dstDir = dst->GetDirection();
//
//    typename SrcType::SpacingType srcSpace = src->GetSpacing();
//    typename DstType::SpacingType dstSpace = dst->GetSpacing();
//
//    unsigned int mindim = min(src->GetImageDimension(), dst->GetImageDimension());
//    unsigned int maxdim = max(src->GetImageDimension(), dst->GetImageDimension());
//
//    std::cerr << "copyInformation" << mindim << " " << maxdim << std::endl;
//    for(unsigned int ii = 0 ; ii < mindim ; ii++) 
//        dstOrigin[ii] = srcOrigin[ii];
//    for(unsigned int ii = mindim ; ii < maxdim ; ii++) 
//        dstOrigin[ii] = 0;
//    
//    for(unsigned int ii = 0 ; ii < mindim ; ii++) 
//        dstSpace[ii] = srcSpace[ii];
//    for(unsigned int ii = mindim ; ii < maxdim ; ii++) 
//        dstSpace[ii] = 1;
//    
//    for(unsigned int ii = 0 ; ii < maxdim ; ii++) {
//        for(unsigned int jj = 0 ; jj < maxdim ; jj++) {
//            if(ii < mindim && jj < mindim) 
//                dstDir(ii,jj) = srcDir(ii,jj);
//            else if(ii == jj)
//                dstDir(ii,jj) = 1;
//            else 
//                dstDir(ii,jj) = 0;
//        }
//    }
//
//    dst->SetSpacing(dstSpace);
//    dst->SetDirection(dstDir);
//    dst->SetOrigin(dstOrigin);
//    std::cerr << "Leaving copyInformation" << mindim << " " << maxdim << std::endl;
//}

const uint32_t SIZE = 5;
struct row
{
    double x[SIZE];
};

int main(int argc, char* argv[])
{
    if(argc != 8) {
        std::cout << "Usage design.txt <outfile> <beta1> <beta2> .... " << std::endl;
        return -1;
    }
    std::cout << "Reading " << argv[3] << std::endl;
    Image3DType::Pointer beta1 = readImage<Image3DType>(argv[2], "");
    std::cout << "Reading " << argv[4] << std::endl;
    Image3DType::Pointer beta2 = readImage<Image3DType>(argv[3], "");
    std::cout << "Reading " << argv[5] << std::endl;
    Image3DType::Pointer beta3 = readImage<Image3DType>(argv[4], "");
    std::cout << "Reading " << argv[6] << std::endl;
    Image3DType::Pointer beta4 = readImage<Image3DType>(argv[5], "");
    std::cout << "Reading " << argv[7] << std::endl;
    Image3DType::Pointer beta5 = readImage<Image3DType>(argv[6], "");

    Image3DType::SizeType size3 = beta1->GetRequestedRegion().GetSize();
    
    std::vector<row> design;
    design.reserve(148);
    
    std::cout << "Reading design from " << argv[1] << std::endl;
    std::ifstream fin(argv[1]);
    std::string tmp;
    getline(fin, tmp);
    while(!fin.eof()) {
        design.resize(design.size()+1);
        std::istringstream iss(tmp);
        for(uint32_t i = 0 ; i < SIZE ; i++) {
            iss >> design.back().x[i];
        }
        getline(fin, tmp);
    }

    Image4DType::Pointer out = Image4DType::New();
    Image4DType::SizeType size4 = {{size3[0], size3[1], size3[2], design.size()}};
    out->SetRegions(size4);
    out->Allocate();
    out->FillBuffer(0);
    itk::ImageLinearIteratorWithIndex<Image3DType> it(beta1, beta1->GetRequestedRegion());
    Image3DType::IndexType index3 = {{0,0,0}};
    Image4DType::IndexType index4;
    double x5,x1,x2,x3,x4;
    for(uint32_t i = 0 ; i < design.size(); i++) {
        for(uint32_t j = 0 ; j < SIZE; j++) {
            std::cout << design[i].x[j] << " ";
        }
        
        it.GoToBegin();
        while(!it.IsAtEnd()) {
            while(!it.IsAtEndOfLine()) {
                if(!isnan(it.Get())) {
                    index3 = it.GetIndex();
                    for(uint32_t l = 0 ; l < 3 ; l++)
                        index4[l] = index3[l];
                    index4[3] = i;
                    x1 = it.Get()*design[i].x[0];
                    x2 = beta2->GetPixel(index3)*design[i].x[1];
                    x3 = beta3->GetPixel(index3)*design[i].x[2];
                    x4 = beta4->GetPixel(index3)*design[i].x[3];
                    x5 = beta5->GetPixel(index3)*design[i].x[4];
                    out->SetPixel(index4, x1+x2+x3+x4+x5);
                }
                ++it;
            }
            it.NextLine();
        }
        std::cout << std::endl;;
    }
    
    copyInformation<Image3DType, Image4DType>(beta1, out);
    writeImage<Image4DType>(argv[2], "", out);

    return 0;
}
