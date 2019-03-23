# system
import os, sys, time, copy, shutil, glob

# plasma
#import sitktools

# numpy, scipy, scikit-learn
import numpy as np
from numpy import random
#import cPickle as pickle
import pickle
import gzip
import SimpleITK as sitk
from PIL import Image
# scipy
import scipy
from scipy.misc import imsave, imread
import scipy.io as sio
import scipy.ndimage as ndimage
from scipy.ndimage import _ni_support
from scipy.ndimage.morphology import distance_transform_edt, binary_erosion, generate_binary_structure
# skimage
from skimage.morphology import label
import pandas as pd

# dice: single label
def dice(seg, gt, val_lb=1):
    """
    ## init
    """
    if seg.shape != gt.shape:
        raise ValueError("Shape mismatch: seg and gt must have the same shape.")
    #
    if val_lb < 0:
        seg = seg > 0
        gt = gt > 0
    else:
        seg = seg == val_lb
        gt = gt == val_lb
    # Compute Dice coefficient
    intersection = np.logical_and(seg, gt)
    return 2. * intersection.sum() / (seg.sum()+gt.sum())

# voe: single label
def voe(seg, gt, val_lb=1):
    """
    ## init
    """
    if seg.shape != gt.shape:
        raise ValueError("Shape mismatch: seg and gt must have the same shape.")
    #
    if val_lb < 0:
        seg = seg > 0
        gt = gt > 0
    else:
        seg = seg == val_lb
        gt = gt == val_lb
    # Compute voe coefficient
    intersection = np.logical_and(seg, gt)
    union = np.logical_or(seg, gt)
    return 100.0*(1.0-np.float32(intersection.sum())/np.float32(union.sum()))

# vd: single label
def vd(seg, gt, val_lb=1):
    """
    ## init
    """
    if seg.shape != gt.shape:
        raise ValueError("Shape mismatch: seg and gt must have the same shape.")
    #
    if val_lb < 0:
        seg = seg > 0
        gt = gt > 0
    else:
        seg = seg == val_lb
        gt = gt == val_lb
    # Compute vd coefficient
    gt = np.int8(gt)
    wori = np.int8(seg - gt)
    return 100.0*(wori.sum()/gt.sum())

"""
## medpy for dists
"""
## basic: surface errors/distances
def surface_distances(result, reference, voxelspacing=None, connectivity=1, iterations=1, ret_all=False):
    """
    # The distances between the surface voxel of binary objects in result and their
    # nearest partner surface voxel of a binary object in reference.
    """
    result = np.atleast_3d(result.astype(np.bool))
    reference = np.atleast_3d(reference.astype(np.bool))
    if voxelspacing is not None:
        voxelspacing = _ni_support._normalize_sequence(voxelspacing, result.ndim)
        voxelspacing = np.asarray(voxelspacing, dtype=np.float64)
        if not voxelspacing.flags.contiguous:
            voxelspacing = voxelspacing.copy()
    # binary structure
    footprint = generate_binary_structure(result.ndim, connectivity)
    # test for emptiness
    if 0 == np.count_nonzero(result):
        raise RuntimeError('The first supplied array does not contain any binary object.')
    if 0 == np.count_nonzero(reference):
        raise RuntimeError('The second supplied array does not contain any binary object.')
    # extract only 1-pixel border line of objects
    result_border = np.logical_xor(result, binary_erosion(result, structure=footprint, iterations=iterations))
    reference_border = np.logical_xor(reference, binary_erosion(reference, structure=footprint, iterations=iterations))
    # compute average surface distance
    # Note: scipys distance transform is calculated only inside the borders of the
    #       foreground objects, therefore the input has to be reversed
    dt = distance_transform_edt(~reference_border, sampling=voxelspacing)
    sds = dt[result_border]
    ##
    if ret_all:
        return sds, dt, result_border, reference_border
    else:
        return sds

## ausdorff Distance.
def hd(result, reference, voxelspacing=None, connectivity=1):
    """
    ## Hausdorff Distance.
    # Computes the (symmetric) Hausdorff Distance (HD) between the binary objects in two
    # images. It is defined as the maximum surface distance between the objects.
    ## Parameters
    # ----------
    # result : array_like
    #     Input data containing objects. Can be any type but will be converted
    #     into binary: background where 0, object everywhere else.
    # reference : array_like
    #     Input data containing objects. Can be any type but will be converted
    #     into binary: background where 0, object everywhere else.
    # voxelspacing : float or sequence of floats, optional
    #     The voxelspacing in a distance unit i.e. spacing of elements
    #     along each dimension. If a sequence, must be of length equal to
    #     the input rank; if a single number, this is used for all axes. If
    #     not specified, a grid spacing of unity is implied.
    # connectivity : int
    #     The neighbourhood/connectivity considered when determining the surface
    #     of the binary objects. This value is passed to
    #     `scipy.ndimage.morphology.generate_binary_structure` and should usually be :math:`> 1`.
    #     Note that the connectivity influences the result in the case of the Hausdorff distance.
    # Returns
    # -------
    # hd : float
    #     The symmetric Hausdorff Distance between the object(s) in ```result``` and the
    #     object(s) in ```reference```. The distance unit is the same as for the spacing of
    #     elements along each dimension, which is usually given in mm.
    #
    # See also
    # --------
    # :func:`assd`
    # :func:`asd`
    # Notes
    # -----
    # This is a real metric. The binary images can therefore be supplied in any order.
    """
    hd1 = surface_distances(result, reference, voxelspacing, connectivity).max()
    hd2 = surface_distances(reference, result, voxelspacing, connectivity).max()
    hd = max(hd1, hd2)
    return hd

## Average surface distance metric.
def asd(result, reference, voxelspacing=None, connectivity=1):
    """
    # Average surface distance metric.
    # Computes the average surface distance (ASD) between the binary objects in two images.
    # Parameters
    # ----------
    # result : array_like
    #     Input data containing objects. Can be any type but will be converted
    #     into binary: background where 0, object everywhere else.
    # reference : array_like
    #     Input data containing objects. Can be any type but will be converted
    #     into binary: background where 0, object everywhere else.
    # voxelspacing : float or sequence of floats, optional
    #     The voxelspacing in a distance unit i.e. spacing of elements
    #     along each dimension. If a sequence, must be of length equal to
    #     the input rank; if a single number, this is used for all axes. If
    #     not specified, a grid spacing of unity is implied.
    # connectivity : int
    #     The neighbourhood/connectivity considered when determining the surface
    #     of the binary objects. This value is passed to
    #     `scipy.ndimage.morphology.generate_binary_structure` and should usually be :math:`> 1`.
    #     The decision on the connectivity is important, as it can influence the results
    #     strongly. If in doubt, leave it as it is.
    # Returns
    # -------
    # asd : float
    #     The average surface distance between the object(s) in ``result`` and the
    #     object(s) in ``reference``. The distance unit is the same as for the spacing
    #     of elements along each dimension, which is usually given in mm.
    # See also
    # --------
    # :func:`assd`
    # :func:`hd`
    # Notes
    # -----
    # This is not a real metric, as it is directed. See `assd` for a real metric of this.
    # The method is implemented making use of distance images and simple binary morphology
    # to achieve high computational speed.
    # Examples
    # --------
    # The `connectivity` determines what pixels/voxels are considered the surface of a
    # binary object. Take the following binary image showing a cross
    #
    # from scipy.ndimage.morphology import generate_binary_structure
    # cross = generate_binary_structure(2, 1)
    # array([[0, 1, 0],
    #        [1, 1, 1],
    #        [0, 1, 0]])
    # With `connectivity` set to `1` a 4-neighbourhood is considered when determining the
    # object surface, resulting in the surface
    # .. code-block:: python
    #
    #     array([[0, 1, 0],
    #            [1, 0, 1],
    #            [0, 1, 0]])
    # Changing `connectivity` to `2`, a 8-neighbourhood is considered and we get:
    # .. code-block:: python
    #
    #     array([[0, 1, 0],
    #            [1, 1, 1],
    #            [0, 1, 0]])
    #
    # , as a diagonal connection does no longer qualifies as valid object surface.
    #
    # This influences the  results `asd` returns. Imagine we want to compute the surface
    # distance of our cross to a cube-like object:
    #
    # cube = generate_binary_structure(2, 1)
    # array([[1, 1, 1],
    #        [1, 1, 1],
    #        [1, 1, 1]])
    #
    # , which surface is, independent of the `connectivity` value set, always
    #
    # .. code-block:: python
    #
    #     array([[1, 1, 1],
    #            [1, 0, 1],
    #            [1, 1, 1]])
    #
    # Using a `connectivity` of `1` we get
    #
    # asd(cross, cube, connectivity=1)
    # 0.0
    #
    # while a value of `2` returns us
    #
    # asd(cross, cube, connectivity=2)
    # 0.20000000000000001
    #
    # due to the center of the cross being considered surface as well.
    """
    sds = surface_distances(result, reference, voxelspacing, connectivity)
    asd = sds.mean()
    return asd

## Average symmetric surface distance.
def assd(result, reference, voxelspacing=None, connectivity=1):
    """
    # Average symmetric surface distance.
    #
    # Computes the average symmetric surface distance (ASD) between the binary objects in
    # two images.
    #
    # Parameters
    # ----------
    # result : array_like
    #     Input data containing objects. Can be any type but will be converted
    #     into binary: background where 0, object everywhere else.
    # reference : array_like
    #     Input data containing objects. Can be any type but will be converted
    #     into binary: background where 0, object everywhere else.
    # voxelspacing : float or sequence of floats, optional
    #     The voxelspacing in a distance unit i.e. spacing of elements
    #     along each dimension. If a sequence, must be of length equal to
    #     the input rank; if a single number, this is used for all axes. If
    #     not specified, a grid spacing of unity is implied.
    # connectivity : int
    #     The neighbourhood/connectivity considered when determining the surface
    #     of the binary objects. This value is passed to
    #     `scipy.ndimage.morphology.generate_binary_structure` and should usually be :math:`> 1`.
    #     The decision on the connectivity is important, as it can influence the results
    #     strongly. If in doubt, leave it as it is.
    #
    # Returns
    # -------
    # assd : float
    #     The average symmetric surface distance between the object(s) in ``result`` and the
    #     object(s) in ``reference``. The distance unit is the same as for the spacing of
    #     elements along each dimension, which is usually given in mm.
    #
    # See also
    # --------
    # :func:`asd`
    # :func:`hd`
    #
    # Notes
    # -----
    # This is a real metric, obtained by calling and averaging
    #
    # >>> asd(result, reference)
    #
    # and
    #
    # >>> asd(reference, result)
    #
    # The binary images can therefore be supplied in any order.
    """
    assd = np.mean((asd(result, reference, voxelspacing, connectivity), asd(reference, result, voxelspacing, connectivity)))
    return assd

"""
## connected comp
"""
## max connected
def max_connected_comp(tmp_buff, lb_num=-1, neighbors=4):
    if lb_num == -1:
        binary_buff = np.uint8(tmp_buff > 0)
    else:
        binary_buff = np.uint8(tmp_buff == lb_num)
    # connected_comp
    connected_group, connected_num = label(binary_buff, neighbors=neighbors, return_num=True)
    comp_sum = []
    for i in range(connected_num+1):
        if i == 0:
            comp_sum.insert(i, 0)
            continue
        comp_sum.insert(i, np.sum(connected_group == i))
    max_comp_ind = np.argmax(comp_sum)
    if lb_num == -1:
        max_comp = np.uint8(connected_group == max_comp_ind)
    else:
        max_comp = np.uint8(connected_group == max_comp_ind) * np.uint8(lb_num)
    #
    return max_comp
## max two connected
def max_two_connected_comp(tmp_buff, lb_num=-1, neighbors=4):
    if lb_num == -1:
        binary_buff = np.uint8(tmp_buff > 0)
    else:
        binary_buff = np.uint8(tmp_buff == lb_num)
    # connected_comp
    connected_group, connected_num = label(binary_buff, neighbors=neighbors, return_num=True)
    comp_sum = []
    for i in range(connected_num+1):
        if i == 0:
            comp_sum.insert(i, 0)
            continue
        comp_sum.insert(i, np.sum(connected_group == i))
    small_to_big_inds = np.argsort(comp_sum)
    first_max = small_to_big_inds[-1]
    second_max = small_to_big_inds[-2]
    if lb_num == -1:
        max_two_comp = np.uint8(connected_group == first_max) + np.uint8(connected_group == second_max)
    else:
        max_two_comp = (np.uint8(connected_group==first_max)+np.uint8(connected_group==second_max)) * np.uint8(lb_num)
    #
    return max_two_comp

header = ("Dice_1","Dice_2", "Dice_1,2", "VOE_1", "VOE_2", "VOE_1,2", "VD_1","VD_2","VD_12","HD_1","HD_2","HD_1,2","ASD_1","ASD_2","ASD_1,2")
rows = list()
subject_ids = list()

CURRDIR="./2019-01-01_ilab3.cs.rutgers.edu/"
CURRDIR=os.path.abspath(CURRDIR)
LR=0.001
os.chdir(CURRDIR)
for i in range(0,2):
    curr=os.path.join(CURRDIR,str(i)+"-time")
    os.chdir(curr)
    for j in range(0,4):
        rows = list()
        subject_ids = list()
        os.chdir(os.path.join(curr,"prediction_"+str(j)+"_"+str(LR)))
        for subject in glob.glob("./*"):
            if not os.path.isdir(subject):
                continue
            subject_ids.append(os.path.basename(subject))
            prediction=sitk.ReadImage(subject+"/prediction.nii.gz")
            truth=sitk.ReadImage(subject+"/truth.nii.gz")
            voxelspacing = list(truth.GetSpacing()).reverse()
            prediction=sitk.GetArrayFromImage(prediction)
            truth=sitk.GetArrayFromImage(truth)
            ##Get Max Connected Map
            prediction_1=max_connected_comp(prediction,lb_num=1)
            prediction_2 = max_connected_comp(prediction, lb_num=2)
            prediction=np.add(prediction_1,prediction_2)
            maxed=sitk.GetImageFromArray(prediction)
            sitk.WriteImage(maxed,subject+"/connected.nii.gz")
            ##Get Seperated Map
            prediction_truth=prediction>0
            prediction_1_truth=prediction_1==1
            prediction_2_truth=prediction_2==2
            truth_1=truth==1
            truth_2=truth==2
            truth_truth=truth>0
            ##Dice
            dice_1=dice(prediction_1_truth,truth_1,1)
            dice_2=dice(prediction_2_truth,truth_2,1)
            dice_12=dice(prediction_truth,truth_truth,1)
            ##VOE
            voe_1=voe(prediction_1_truth,truth_1,1)
            voe_2=voe(prediction_2_truth,truth_2,1)
            voe_12=voe(prediction_truth,truth_truth,1)
            ##VD
            vd_1=vd(prediction_1_truth,truth_1,1)
            vd_2=vd(prediction_2_truth,truth_2,1)
            vd_12=vd(prediction_truth,truth_truth,1)
            hd_1=hd(prediction_1_truth,truth_1,voxelspacing)
            hd_2=hd(prediction_2_truth,truth_2,voxelspacing)
            hd_12=hd(prediction_truth,truth_truth,voxelspacing)
            asd_1=asd(prediction_1_truth,truth_1,voxelspacing)
            asd_2=asd(prediction_2_truth,truth_2,voxelspacing)
            asd_12=asd(prediction_truth,truth_truth,voxelspacing)
            rows.append([dice_1,dice_2,dice_12,voe_1,voe_2,voe_12,vd_1,vd_2,vd_12,hd_1,hd_2,hd_12,asd_1,asd_2,asd_12])
        df=pd.DataFrame.from_records(rows, columns=header, index=subject_ids)
        df.loc['avg'] = df.mean()
        df.to_csv("../../statistics-"+str(i)+"_"+str(j)+".csv")
