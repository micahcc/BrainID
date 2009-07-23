#include <itkImageFileWriter.h>
#include <itkImageFileReader.h>
#include <cstdio>
#include <sys/stat.h>

#include "modNiftiImageIO.h"
#include "tools.h"


int main(int argc, char* argv[])
{
    if(argc != 5) {
        printf("Usage:\n%s <input> <output> <start> <stop>\n", 
                    argv[0]);
        exit(1);
    }
    
  // Attempt to get the file attributes
    struct stat stFileInfo;
    int status;
    status = stat(argv[2],&stFileInfo);
    if(status == 0) {
        fprintf(stderr, "File Exists: %s\n", argv[2]);
        exit(2);
    }

    itk::ImageFileReader< itk::OrientedImage< double, 4> >::Pointer  reader = 
                itk::ImageFileReader< itk::OrientedImage< double, 4> >::New();
    reader->SetImageIO(itk::modNiftiImageIO::New());
    std::vector< itk::OrientedImage< double, 4>::Pointer > images;
    reader->SetFileName( argv[1] );
    reader->Update();
    
    itk::ImageFileWriter< itk::OrientedImage< double, 4> >::Pointer  writer = 
                itk::ImageFileWriter< itk::OrientedImage< double, 4> >::New();
    writer->SetImageIO(itk::modNiftiImageIO::New());
    writer->SetFileName(argv[2]);
    
    itk::OrientedImage< double, 4 >::Pointer result = prune<double>(
                reader->GetOutput(), 3, atoi(argv[3]), atoi(argv[4]));
    writer->SetInput( result );
    writer->Update();

    return 0;
}

