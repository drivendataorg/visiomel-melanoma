import numpy as np
from PIL import Image

import os
import glob
import pandas as pd

import matplotlib.pyplot as plt

from scipy import ndimage as ndi
import skimage.filters as sk_filters
from skimage.util import random_noise
from skimage import feature
from skimage.morphology import area_closing, convex_hull_object, remove_small_holes, remove_small_objects,label

import pyvips


import torch
import torch.nn as nn
import torch.nn.functional as F


####################################################
########  Util functions from deephistopath ########
####################################################
def pil_to_np_rgb(pil_img):
  """
  Convert a PIL Image to a NumPy array.

  Note that RGB PIL (w, h) -> NumPy (h, w, 3).

  Args:
    pil_img: The PIL Image.

  Returns:
    The PIL image converted to a NumPy array.
  """
  rgb = np.asarray(pil_img)
  return rgb

def mask_rgb(rgb, mask):
  """
  Apply a binary (T/F, 1/0) mask to a 3-channel RGB image and output the result.

  Args:
    rgb: RGB image as a NumPy array.
    mask: An image mask to determine which pixels in the original image should be displayed.

  Returns:
    NumPy array representing an RGB image with mask applied.
  """
  result = rgb * np.dstack([mask, mask, mask])
  return result
def filter_rgb_to_grayscale(np_img, output_type="uint8"):
  """
  Convert an RGB NumPy array to a grayscale NumPy array.

  Shape (h, w, c) to (h, w).

  Args:
    np_img: RGB Image as a NumPy array.
    output_type: Type of array to return (float or uint8)

  Returns:
    Grayscale image as NumPy array with shape (h, w).
  """
  # Another common RGB ratio possibility: [0.299, 0.587, 0.114]
  grayscale = np.dot(np_img[..., :3], [0.2125, 0.7154, 0.0721])
  if output_type != "float":
    grayscale = grayscale.astype("uint8")
  return grayscale
def filter_complement(np_img, output_type="uint8"):
  """
  Obtain the complement of an image as a NumPy array.

  Args:
    np_img: Image as a NumPy array.
    type: Type of array to return (float or uint8).

  Returns:
    Complement image as Numpy array.
  """
  if output_type == "float":
    complement = 1.0 - np_img
  else:
    complement = 255 - np_img
  return complement
def filter_otsu_threshold(np_img, output_type="uint8"):
  """
  Compute Otsu threshold on image as a NumPy array and return binary image based on pixels above threshold.

  Args:
    np_img: Image as a NumPy array.
    output_type: Type of array to return (bool, float, or uint8).

  Returns:
    NumPy array (bool, float, or uint8) where True, 1.0, and 255 represent a pixel above Otsu threshold.
  """
  otsu_thresh_value = sk_filters.threshold_otsu(np_img)
  # mask = np.mean(np_img, axis=2) >
  otsu = (np_img > otsu_thresh_value)
  if output_type == "bool":
    pass
  elif output_type == "float":
    otsu = otsu.astype(float)
  else:
    otsu = otsu.astype("uint8") * 255
  return otsu


####################################################
#################  Our Transforms ##################
####################################################
class PIL_mask(object):
    def __init__(self, min_remove=None, return_hull=False, return_mask=False):
        """
        Args:
            min_remove (float, optional): Float in [0,1], threshold on percentage of removed pixels. 
                                          If threshold is not met, return full image.
                                        Defaults to None.
            return_hull (bool, optional): Return the Convex Hull, otherwise returns antepenultimate masking step.
                                         Defaults to False.
            return_hull (bool, optional): Return the mask instead of masked image..
                                         c'est le cas ou on se rabat sur masquer le Gradcam et pas l'image en input
                                        Defaults to False.
        """
        self.min_remove = min_remove
        self.return_hull = return_hull
        self.return_mask = return_mask

    def __call__(self, pil_image):
        ## Rotate images if image is in width :
        # if pil_image.size[0] > pil_image.size[1]:
        #     img = img.rotate(90, Image.NEAREST, expand = 1)
        rgb = pil_to_np_rgb(pil_image)

        # Define custom filters :
        bubble_thresh = 660
        num_white = np.sum(np.sum(rgb, axis=2) > 700)
        if num_white < 100 :
            bubble_thresh = 600
        elif num_white > 600000:
            bubble_thresh = 400
        filter_bubble = np.sum(rgb, axis=2) < bubble_thresh
        filter_black = np.sum(rgb, axis=2) > 250
        filter_same = 1 - ((rgb[:,:,0] == rgb[:,:,1]) * (rgb[:,:,1] == rgb[:,:,2]) * (rgb[:,:,0] == rgb[:,:,2]))
        filter_almost_same =  ((np.abs(rgb[:,:,0] - rgb[:,:,1]) > 1) 
                                * (np.abs(rgb[:,:,1] - rgb[:,:,2]) > 1)
                                * (np.abs(rgb[:,:,0] - rgb[:,:,2]) > 1))
        # Apply Custom Filters
        masked_rgb = mask_rgb(rgb, 1- filter_bubble*filter_black*filter_same*filter_almost_same)
        
        # Apply Library filters
        complement_grayscale = filter_complement(filter_rgb_to_grayscale(masked_rgb))
        otsu_mask = filter_otsu_threshold(complement_grayscale, output_type="bool")
        
        # Remove small holes
        no_holes_mask = remove_small_holes(otsu_mask, area_threshold=400)
        
        # Remove small objects
        rm_small_mask = remove_small_objects(no_holes_mask, min_size=400)
        rm_small_masked = mask_rgb(rgb, rm_small_mask)

        # Get convex hull
        hull_mask = convex_hull_object(rm_small_mask)
        hull_masked = mask_rgb(rgb, hull_mask)
        
        final = hull_masked
        final_mask = hull_mask
        # Retun Hull only if specified
        if not self.return_hull :
            final = rm_small_masked
            final_mask = rm_small_mask
        # Only apply pre_processing if removing enough values?
        if self.min_remove is not None:
            kept = np.sum(final_mask)
            removed = np.sum(1-final_mask)
            ratio =kept/(kept+removed)
            if ratio > self.min_remove :
                final = rgb
        if self.return_mask:
            final = final_mask
        return final
    
class toPatches(object):
    def __init__(self, num_patches=10, patch_size=512,replacement=False):
        """
        Args:
            num_patches (int, optional): number of patches to sample.
            patch_size (int or tuple of ints, optional): Size of patches to sample
            replacement (bool): Wether to replace when randomly sampling. Should be False if not debugging.
        """
        self.num_patches = num_patches
        self.patch_size = (patch_size,patch_size) if isinstance(patch_size, int) else patch_size
        self.replacement = replacement
        
    def __call__(self, slide, heatmap):
        """
            slide: (high res) tiff thing (C,H,W) from which to extract patches.
            heatmap: (low res) batched torch Tensor (1,H,W) giving scores of each pixel.
        """
        
        assert (heatmap.dim() == 3) and (heatmap.shape[0] == 1), f"Expected heatmap of shape (1,h,w), got {heatmap.shape}"
        
        # Probably quicker and lighter to get indexes to slice in low res, rather than upsample heatmap..
        factor = slide.width//heatmap.shape[1] #num_upsamples = slide.width//heatmap.shape[1]
#         print(num_upsamples)
#         factor = 2**num_upsamples
        h_patch,w_patch = tuple(s//factor for s in self.patch_size)
    
        # Need to eventually crop so we can perfectly tile image:
        _,h_heatmap,w_heatmap = heatmap.shape
        heatmap = heatmap[:, :-(h_heatmap%h_patch), :-(w_heatmap%w_patch)]
        
        # get cumulative sum of heatmap patches
        kernel = torch.ones(1,1,h_patch,w_patch)
        probabilities = F.conv2d(heatmap, kernel, stride=(h_patch,w_patch))

        indexes = torch.multinomial(probabilities.flatten(start_dim=1), self.num_patches,replacement=self.replacement)

        # Convert indexes into tuple of coordnates:
        rows = (indexes // probabilities.shape[2]) * factor * (heatmap.shape[1]//probabilities.shape[1])
        cols = (indexes % probabilities.shape[2]) * factor  * (heatmap.shape[2]//probabilities.shape[2])

        patch_list = []
        for (i,j) in zip(rows[0],cols[0]):
            region = slide.crop(int(i), int(j), self.patch_size[0], self.patch_size[1])
            patch = np.ndarray(buffer=region.write_to_memory(),
                            dtype=np.uint8,
                            shape=(region.height, region.width, region.bands))
            patch_list.append(patch)
        return patch_list


if __name__ == "__main__":
   # Example usage
    heatmap = torch.zeros(1,868,452)
    heatmap[0,868//2,452//2] = 100

    slide = pyvips.Image.new_from_file("data2/94c4bcxq.tif", page=0)

    to_patch = toPatches(num_patches=10, patch_size=1024, replacement=True) # In real usage remove replacement=True...
    res = to_patch(slide, heatmap)
    fig, axs =plt.subplots(1,10, figsize=(20,8))
    for i in range(len(axs)):
        axs[i].imshow(res[i])
    plt.show()