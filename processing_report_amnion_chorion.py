from pprint import pprint
from pathlib import Path
from PIL import Image


path_dataset = Path(r"Z:\0-Projects and Experiments\KS - OCT membranes\human_dataset_copy_no_oct_files")

# get list of all subsamples
list_subsample_dirs = list(path_dataset.glob("*"))
list_subsample_dirs = [p for p in list_subsample_dirs if p.is_dir()]


count_samples_to_process = 0
count_total = 0
for pos, dir_subsample in enumerate(list_subsample_dirs[:]):
    pass
    print(f"{pos})  {dir_subsample.stem}")
    # get pressure files
    
    list_layers = ["amnion", "chorion"]
    # layer = "amnion"
    for layer in list_layers:
        pass
        list_path_pressure_files = list(dir_subsample.rglob(f"*_{layer}_*.txt")) + \
                                        list(dir_subsample.rglob(f"_{layer}_*Mode2D*.txt"))
        
        for idx, path_pressure in enumerate(list_path_pressure_files):
            pass
            
            # one of the pressure files with _Pressure.txt or Pressure_failure.txt
            if "ressure" in str(path_pressure):
                base_name = path_pressure.stem.rsplit("_", 1)[0]
                path_im = path_pressure.parent / f"{base_name}.tiff"
            # one of the pressure files with Mode2D.txt
            else:
                base_name = path_pressure.stem
                path_im = path_pressure.parent / f"{base_name}.tiff"
            
            # If corresponding tiff file doesn't exist continue to next sample
            if not path_im.exists():
                continue

           

            # valid sample to process
            count_total += 1
            
            # check if segmentation and apex rise exist
            bool_seg_tiff_file_exists = Path(path_im.parent / (path_im.stem + "_seg.tiff")).exists()
            bool_seg_file_exists = Path(path_im.parent / (path_im.stem + "_seg.csv")).exists()
            bool_apex_rise_file_exists = Path(path_im.parent / (path_im.stem + "_Pressure_Apex.csv")).exists()
            
            
            # if bool_seg_tiff_file_exists and bool_seg_file_exists and bool_apex_rise_file_exists:
            if bool_apex_rise_file_exists:

                print(f"{' ' * 8} [x] {layer} | {path_pressure.parent.stem}")
            elif (path_im.parent / "reg.txt").exists() : ## registration problem
                print(f"{' ' * 8} [ - ] {layer} | {path_pressure.parent.stem} | registration_error")
            
            elif (path_im.parent / "flip.txt").exists() : ## registration problem
                print(f"{' ' * 8} [ & ] {layer} | {path_pressure.parent.stem} | flipped frames")
            
            elif (path_im.parent / "rename.txt").exists() : ## registration problem
                print(f"{' ' * 8} [ R ] {layer} | {path_pressure.parent.stem} | rename files")
                 
            elif (path_im.parent / "bad_press.txt").exists() : 
                print(f"{' ' * 8} [ R ] {layer} | {path_pressure.parent.stem} | bad pressure file")
            elif (path_im.parent / "fail.txt").exists() :
                print(f"{' ' * 8} [ F ] {layer} | {path_pressure.parent.stem} | pressure file of failure")
            elif (path_im.parent / "size_limit.txt").exists() :
                print(f"{' ' * 8} [ F ] {layer} | {path_pressure.parent.stem} | file size limit reached, incomplete image stack")
                         
            else:
                count_samples_to_process += 1
                print(f"{' ' * 8} [ ] {layer} | {path_pressure.parent.stem}")
                # if not bool_seg_tiff_file_exists : print(f"{' ' * 12} [ ] segmentation tiff" ) 
                # if not bool_seg_file_exists: print(f"{' ' * 12} [ ] thickness/length/area csv" ) 
                if not bool_apex_rise_file_exists: print(f"{' ' * 12} [ ] apex rise csv (matlab) " ) 

print(f"samples to process: {count_samples_to_process}/{count_total}")