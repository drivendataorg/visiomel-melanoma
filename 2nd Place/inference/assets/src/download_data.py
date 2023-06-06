#### 
####
#### Usage : python3 download_data.py (--min_size 1e6)
####
#### Note: data/ folder should exist and contain csv clinical data sheet. 
####
import warnings
import subprocess
import os
import argparse
import numpy as np
import pandas as pd

from tqdm import tqdm
import pyvips


##### Eventually parse argument
parser = argparse.ArgumentParser()
parser.add_argument('--min_size', type=float, default=1.3e6, help='Minimum resolution to keep')
parser.add_argument('-v', "--verbose", type=bool, default=False, help='Wether to display aws outputs')
args = parser.parse_args()


#####
root = os.getcwd()
print(root)
if not os.path.exists(root + "/data/images/"):
    os.makedirs(root + "/data/images/")

####################################################################
# Get all image names
list_names = pd.read_csv(root+"/data/train_metadata.csv").filename.to_numpy()
list_buckets = pd.read_csv(root + "/data/train_metadata.csv").eu_tif_url.to_numpy()

# Initialize lists for new dataframe containing image sizes
num_pages, kept_page,  init_size_width, init_size_height = [],[],[],[]

for (name, bucket) in tqdm(zip(list_names,list_buckets), total=len(list_names)):#, position=0,leave=True):
    # TODO : download image in "data/images/" folder
    
    if args.verbose:
        subprocess.run(["aws", "s3", "cp", bucket,  root+ "/data/images/", "--no-sign-request"])#, capture_output=True, text=True)
    else:
        subprocess.run(["aws", "s3", "cp", bucket,  root+ "/data/images/", "--no-sign-request"], capture_output=True, text=True)
    # os.system("aws s3 cp "+ bucket+ " "+ root+ "/data/images/ --no-sign-request")

    ### Append info for dataframe: ###
    # Num pages:
    slide = pyvips.Image.new_from_file(os.path.join(root +"/data/images/",name))#, page=0)
    n = slide.get_n_pages()
    num_pages.append(n)
    # Height and width of page 0:
    page = 0
    slide = pyvips.Image.new_from_file(os.path.join(root+"/data/images/",name), page=page)

    init_size_width.append(slide.width)
    init_size_height.append(slide.height)
    size = init_size_width[-1] * init_size_height[-1]
    ### Decide which page to keep : ###
    while (size > args.min_size) :
        if page > n : # On sait jamais avec ces fous...
            size = -1
            page = -1
            break
        page += 1
        slide = pyvips.Image.new_from_file(os.path.join(root +"/data/images/",name), page=page)
        size = slide.width*slide.height
    kept_page.append(page)

    # Save page as png :
    slide.write_to_file(root +"/data/images/"+name[:-4]+".png")
    # Delete tiff file :
    os.remove(root +"/data/images/" + name)
    

if len(list_names != len(num_pages)):
    warnings.warn(f"Potential problem : {len(list_names)=} {len(num_pages)=} {len(kept_page)=} {len(init_size_width)=} {len(init_size_height)=}")
    # Create and save dataframe :
    df = pd.DataFrame(data={
        "filename":list_names[:len(num_pages)],
        "num_pages":num_pages,
        "kept_page": kept_page,
        "init_size_width":init_size_width,
        "init_size_height":init_size_height
    })
else :
    df = pd.DataFrame(data={
        "filename":list_names,
        "num_pages":num_pages,
        "kept_page": kept_page,
        "init_size_width":init_size_width,
        "init_size_height":init_size_height
    })
df.to_csv(root +"/data/tiff_info_data.csv")
