#include <itkImageFileWriter.h>
#include <itkImageFileReader.h>
#include <cstdio>
#include <sys/stat.h>

#include "modNiftiImageIO.h"
#include "tools.h"


int main(int argc, char* argv[])
{
    if(argc < 4) {
        printf("Usage:\n%s <output> <list of files in order to be attached>\n", 
                    argv[0]);
        exit(1);
    }
    
  // Attempt to get the file attributes
    struct stat stFileInfo;
    int status;
    status = stat(argv[1],&stFileInfo);
    if(status == 0) {
        fprintf(stderr, "File Exists: %s\n", argv[1]);
        exit(2);
    }

    std::vector< itk::OrientedImage< double, 4>::Pointer > images;
    for(int ii = 2 ; ii < argc ; ii++) {
        itk::ImageFileReader< itk::OrientedImage< double, 4> >::Pointer  reader = 
                    itk::ImageFileReader< itk::OrientedImage< double, 4> >::New();
        reader->SetImageIO(itk::modNiftiImageIO::New());
        printf("%i: %s\n", ii-2, argv[ii]);
        reader->SetFileName( argv[ii] );
        reader->Update();
        images.push_back(reader->GetOutput());
    }
    
    itk::ImageFileWriter< itk::OrientedImage< double, 4> >::Pointer  writer = 
                itk::ImageFileWriter< itk::OrientedImage< double, 4> >::New();
    writer->SetImageIO(itk::modNiftiImageIO::New());
    writer->SetFileName(argv[1]);
    
    itk::OrientedImage< double, 4 >::Pointer result = concat<double>(images, 3);
    writer->SetInput( result );
    writer->Update();

    return 0;
}
