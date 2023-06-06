import numpy as np, pandas as pd
import glob,os,sys,shutil,gc,copy,math, warnings,random,string,logging,multiprocessing,subprocess,time
import PIL,pyvips

import skimage.io as sk 
from datetime import timedelta

from wsi import slide,filters,tiles,util

START_TIME = time.time()
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger()
PIL.Image.MAX_IMAGE_PIXELS = None

DATA_ROOT = '/code_execution/data/'
RAW_IMG_DIR = DATA_ROOT #'./'

BASE_PAGE = 5
STAGE = 'test'
slide.RAW_IMG_DIR = RAW_IMG_DIR
slide.BASE_PAGE = BASE_PAGE


meta = pd.read_csv(f"{DATA_ROOT}/test_metadata.csv")

PAGE_IX_MULS ={i:2**(BASE_PAGE-i) for i in range(BASE_PAGE+1)}
DIR_OUTPUT_TILES = f'./workspace/tiles/{STAGE}' #args.stage


SCORE_THRESH=0.1;MAX_TILES_PER_PAGE=10000;N_MIN_ERROR_TILES=0 
PAGES_TO_EXTRACT = {}
PAGES_TO_EXTRACT[56] = [3] 

    
RANDOM_STATE = 41
def fix_seed(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
fix_seed(RANDOM_STATE)



def save_tiles_for_page(cur_page,base_sz,slide_name,df_tiles):
    patch_size = PATCH_SIZES_ACT[cur_page]
    slide_img = pyvips.Image.new_from_file(f'{RAW_IMG_DIR}/{slide_name}.tif', page=cur_page)
    RES_MUL = PAGE_IX_MULS[cur_page] #2**(base_page-cur_page)
    
    dir_output = f'{DIR_OUTPUT_TILES}/{base_sz}/{cur_page}_{patch_size}/{slide_name}' #b,p,sz
    dir_output_img = f'{dir_output}/img' #b,p,sz
    os.makedirs(dir_output_img,exist_ok=True)
    ds_tiles=[]
    save_ctr=0
    for idx, row in df_tiles.iterrows():
        if save_ctr==MAX_TILES_PER_PAGE: ##generated maximum tiles for page, exit
            break
        y = row['Row Start']
        x = row['Col Start']

        if (y<0 or x<0):
            #print(f"skipping: {slide_name}, bad coords x:{x} y:{y}")  
            continue
        
        
        x1 = x*RES_MUL
        y1 = y*RES_MUL
        
        region_width = region_height = patch_size #PATCH_SIZES_ACT[cur_page]
        if x1 + region_width >slide_img.width:
            _diff=slide_img.width-(x1+region_width)
            region_width = slide_img.width - x1
            if len(df_tiles)>N_MIN_ERROR_TILES:
                #print(f'skipping {slide_name} since {x1} + {region_width} >{slide_img.width} by {_diff}, new width {region_width}')
                continue
        if y1 + region_height >slide_img.height:
            _diff=slide_img.height-(y1+region_height)
            region_height = slide_img.height - y1
            if len(df_tiles)>N_MIN_ERROR_TILES:
                #print(f'skipping {slide_name} since {y1} + {region_height} >{slide_img.height} by {_diff} new height {region_height}')
                continue
                
        try:
            region = pyvips.Region.new(slide_img).fetch(x1, y1, region_width, region_height)
            img = np.ndarray(
                buffer=region,
                dtype=np.uint8,
                shape=(region_height, region_width, 3)) #rgb image
            
            img = PIL.Image.fromarray(img)
            img.save(f'{dir_output_img}/{row.tile_id}_{y1}_{x1}.jpeg', quality=90)
            save_ctr+=1

            row['w']=region_width
            row['h']=region_height
            row['swidth']=slide_img.width
            row['sheight']=slide_img.height
            
            ds_tiles.append(row)
            
        except Exception as ex:
            #print(f'Failed for {slide_name}. x: {x}, y: {y} x1: {x1}, y1: {y1} reg_w: {region_width}, reg_h: {region_height} ')
            #print(f'slide width: {slide_img.width} height: {slide_img.height}  cur_page: {cur_page}' )
            print(ex)
        
    d = pd.DataFrame(ds_tiles)
    d.to_csv(f'{dir_output}/tile_meta.csv',index=False)

def generate_tiles_for_slide_list(slide_names,base_sz,pages_to_extract):
    for slide_name in slide_names:
        # ##generate tiles
        df = pd.read_csv(f'{slide.TILE_DATA_DIR}/{slide_name}-tile_data.csv',skiprows=14).sort_values(by='Score',ascending=False)
        df['og_ntiles'] = len(df)
        df1=df[df.Score>SCORE_THRESH]
          #filter scores
        if len(df1)>0:
            df = df1
        else:
            pass #print(f'Ignoring Score: {slide_name}')
            
        df = df.reset_index(drop=True)
        df['tile_id'] = df.index
        df['slide_id'] = slide_name
        
        #df['filename'] = df['slide_id'] + '.tif'
        for page in pages_to_extract:
            save_tiles_for_page(page,base_sz,slide_name,df)
        #gen_tiles(RAW_IMG_DIR,base_sz,df,pages_to_extract)


def multiprocess_generate_tiles(slides_list,base_sz,pages_to_extract):
    num_slides = len(slides_list)

    num_processes = min(multiprocessing.cpu_count(),5)
    pool = multiprocessing.Pool(num_processes)

    if num_processes > num_slides:
        num_processes = num_slides
    
    slides_per_process = num_slides / num_processes
    tasks = []
    for num_process in range(1, num_processes + 1):
        start_index = (num_process - 1) * slides_per_process + 1
        end_index = num_process * slides_per_process
        start_index = int(start_index)
        end_index = int(end_index)
        sublist = slides_list[start_index - 1:end_index]
        tasks.append((sublist,base_sz,pages_to_extract))
        #print(f"Task # {num_process} Process slides {sublist}")
    
  # start tasks
    results = []
    for t in tasks:
        results.append(pool.apply_async(generate_tiles_for_slide_list, t))

    for result in results:
        _ = result.get()


DATA_ROOT = '/code_execution/data/'
meta = pd.read_csv(f"{DATA_ROOT}/test_metadata.csv")
NAMES = [n.split('.')[0] for n in meta.filename.values]


BASE_SZ = 56
tiles.TILE_SIZE_BASE = BASE_SZ
slide.TILE_DATA_DIR = os.path.join(slide.BASE_DIR, f"tile_data/{BASE_SZ}")
slide.TOP_TILES_DIR = os.path.join(slide.BASE_DIR, f"top_tiles/{BASE_SZ}")
PATCH_SIZES_ACT ={i:BASE_SZ*2**(BASE_PAGE-i) for i in range(BASE_PAGE)} #patch size to extract for each page

logger.info(f'********* GENERATING TILE META {BASE_SZ} tile sizes: {PATCH_SIZES_ACT} **********')
tiles.multiprocess_filtered_images_to_tiles(image_list=NAMES, display=False, save_summary=False, save_data=True, save_top_tiles=False)
pg=PAGES_TO_EXTRACT[BASE_SZ][0]
sz=PATCH_SIZES_ACT[pg]
logger.info(f'********* GENERATING TILES {BASE_SZ}_{pg}_{sz} **********')
multiprocess_generate_tiles(NAMES,BASE_SZ,pages_to_extract=PAGES_TO_EXTRACT[BASE_SZ])

elapsed = time.time() - START_TIME
logger.info(f'######### DONE GENERATING TILES {BASE_SZ}_{pg}_{sz} ######## TOTAL TIME: {timedelta(seconds=elapsed)}')
gc.collect()

