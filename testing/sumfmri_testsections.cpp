
///////////////////////////////////////////////////
//testing code, write out an image for each section
void test_sections(Image4DType::Pointer fmri_img, std::string filename,
            std::list<SectionType*>& active_voxels) 
{
    fprintf(stderr, "Writing out every section, from active voxel list\n");
    // writer
    Image4DType::RegionType fmri_region = fmri_img->GetRequestedRegion();

    itk::ImageFileWriter< Image3DType >::Pointer writer = 
        itk::ImageFileWriter< Image3DType >::New();

    Image3DType::Pointer outputImage = Image3DType::New();

    Image3DType::RegionType out_region;
    Image3DType::IndexType out_index;
    Image3DType::SizeType out_size;
    out_size[0] = fmri_region.GetSize()[0];
    out_size[1] = fmri_region.GetSize()[1];
    out_size[2] = fmri_region.GetSize()[2];

    out_index[0] = fmri_region.GetIndex()[0];
    out_index[1] = fmri_region.GetIndex()[1];
    out_index[2] = fmri_region.GetIndex()[2];

    out_region.SetSize(out_size);
    out_region.SetIndex(out_index);

    outputImage->SetRegions( out_region );
    //outputImage->CopyInformation( fmri_img );
    outputImage->Allocate();

    itk::ImageSliceIteratorWithIndex<Image3DType> 
        out_it(outputImage, outputImage->GetRequestedRegion());
    out_it.SetFirstDirection(0);
    out_it.SetSecondDirection(1);

    std::ostringstream os;
    std::list< SectionType* >::iterator section_it = active_voxels.begin();
    std::list<SliceIterator4D>::iterator voxel_it;
    while(section_it != active_voxels.end()) {
        out_it.GoToBegin();
        while(!out_it.IsAtEnd()) {
            fprintf(stderr, ".");
            while(!out_it.IsAtEndOfSlice()) {
                while(!out_it.IsAtEndOfLine()) {
                    out_it.Value() = 0;
                    ++out_it;
                }
                out_it.NextLine();
            }
            out_it.NextSlice();
        }

        voxel_it = (*section_it)->list.begin();
        fprintf(stdout, "Label: %u Number: %u\n", (*section_it)->label, 
                (*section_it)->list.size());
        while(voxel_it != (*section_it)->list.end()) {
            out_index[0] = voxel_it->GetIndex()[0];
            out_index[1] = voxel_it->GetIndex()[1];
            out_index[2] = voxel_it->GetIndex()[2];
            fprintf(stdout, "%li %li %li\n", out_index[0],
                    out_index[1], out_index[2]);
            out_it.SetIndex(out_index);
            out_it.Value() = voxel_it->Get();
            voxel_it++;
        }

        os.str("");
        os << filename << (*section_it)->label << ".nii.gz";
        fprintf(stderr, "writing: %s\n", os.str().c_str());
        writer->SetFileName( os.str() );  
        writer->SetInput(outputImage);
        writer->Update();

        section_it++;
    }
}
