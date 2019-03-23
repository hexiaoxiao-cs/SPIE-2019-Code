import gc
import glob
import os

import SimpleITK as sitk
import numpy as np
from skimage import exposure


# bias correction on a sitk image data. from: 
def mriN4BiasCorrection_sitkImage(input_sitk, numOfIters=10, numOfFilltingLev=4):
    # get mask
    mask_sitk = sitk.OtsuThreshold(input_sitk, 0, 1, 200)
    # cast to float32
    input_sitk = sitk.Cast(input_sitk, sitk.sitkFloat32)
    # correction
    corrector = sitk.N4BiasFieldCorrectionImageFilter()
    # set max num of iterations
    numberFilltingLevels = 4
    if numOfFilltingLev > 0:
        numberFilltingLevels = numOfFilltingLev
    corrector.SetMaximumNumberOfIterations([int(numOfIters)] * numberFilltingLevels)
    # do
    print(input_sitk.GetSpacing())
    output_sitk = corrector.Execute(input_sitk, mask_sitk)
    return output_sitk


def get_image(subject_folder, name):
    file_card = os.path.join(subject_folder, "*" + name + ".nii.gz")
    try:
        return glob.glob(file_card)[0]
    except IndexError:
        raise RuntimeError("Could not find file matching {}".format(file_card))


def resize(image, method=sitk.sitkLinear):
    resize_xyz = 1.5
    method = sitk.sitkLinear
    img = image
    # we rotate the image according to its transformation using the direction and according to the final spacing we want
    factor = np.asarray(img.GetSpacing()) / resize_xyz
    factorSize = np.asarray(img.GetSize() * factor, dtype=float)
    newSize = factorSize
    newSize = newSize.astype(dtype=int)
    newSize = newSize.tolist()
    resampler = sitk.ResampleImageFilter()
    resampler.SetReferenceImage(img)
    resampler.SetOutputSpacing([resize_xyz, resize_xyz, resize_xyz])
    resampler.SetSize(newSize)
    resampler.SetInterpolator(method)
    imgResampled = resampler.Execute(img)
    return imgResampled


# convert buffer as a sitk image
def convBufferToItkImage(buffer, orgImage):
    newImage = sitk.GetImageFromArray(buffer)
    newImage.SetOrigin(orgImage.GetOrigin())
    newImage.SetDirection(orgImage.GetDirection())
    newImage.SetSpacing(orgImage.GetSpacing())
    return newImage


#
def resamplingImage_Sp2(image_sitk, sp2=[1.0, 1.0, 1.0], sampling_way=sitk.sitkNearestNeighbor,
                        sampling_pixel_t=sitk.sitkUInt16):
    ## get org info
    buff_sz1, sp1, origin = image_sitk.GetSize(), image_sitk.GetSpacing(), image_sitk.GetOrigin()
    direction = image_sitk.GetDirection()
    # rate
    fScales = np.zeros(len(sp1), np.float32)
    for i in range(len(sp1)):
        fScales[i] = np.float32(sp2[i]) / np.float32(sp1[i])
    # change buff size
    buff_sz2 = list()
    for i in range(len(buff_sz1)):
        buff_sz2.append(int(np.round(buff_sz1[i] / fScales[i])))
    # resampled info
    print("Orig Size ", buff_sz1, "\nNew Size ", buff_sz2)
    print("Orig Sp ", sp1, "\nNew Sp ", sp2)
    print(origin)
    ## resample
    t = sitk.Transform(3, sitk.sitkScale)
    t.SetParameters((1.0, 1.0, 1.0))
    resampled_image_sitk = sitk.Resample(image_sitk, buff_sz2, t, sampling_way, origin, sp2, direction, 0.0,
                                         sampling_pixel_t)
    print("New Image size:", resampled_image_sitk.GetSize())
    return resampled_image_sitk


def otsu_filter(image):
    otsu_filter = sitk.OtsuThresholdImageFilter()
    seg = otsu_filter.Execute(image)
    return seg


def hist_match(source, template):
    """
    Adjust the pixel values of a grayscale image such that its histogram
    matches that of a target image
    Arguments:
    -----------
        source: np.ndarray
            Image to transform; the histogram is computed over the flattened
            array
        template: np.ndarray
            Template image; can have different dimensions to source
    Returns:
    -----------
        matched: np.ndarray
            The transformed output image
    """

    oldshape = source.shape
    source = source.ravel()
    template = template.ravel()

    # get the set of unique pixel values and their corresponding indices and
    # counts
    s_values, bin_idx, s_counts = np.unique(source, return_inverse=True,
                                            return_counts=True)
    t_values, t_counts = np.unique(template, return_counts=True)

    # take the cumsum of the counts and normalize by the number of pixels to
    # get the empirical cumulative distribution functions for the source and
    # template images (maps pixel value --> quantile)
    s_quantiles = np.cumsum(s_counts).astype(np.float64)
    s_quantiles /= s_quantiles[-1]
    t_quantiles = np.cumsum(t_counts).astype(np.float64)
    t_quantiles /= t_quantiles[-1]

    # interpolate linearly to find the pixel values in the template image
    # that correspond most closely to the quantiles in the source image
    # interp_t_values = np.zeros_like(source,dtype=float)
    interp_t_values = np.interp(s_quantiles, t_quantiles, t_values)

    return interp_t_values[bin_idx].reshape(oldshape)


def getTargetCenter_fromBuffL(buff_l):
    """
    ## inputs: buff_l from an itk image of label, kval is value of target
    ## outputs: target center in voxel coordinates
    """
    z, y, x = np.where(buff_l > 0)

    # center in voxel coordinates, z, x, and y are in voxel coordinates
    # len_axial = [1] --> y
    # len_coronal = [0] --> z
    # len_sagittal = [2] --> x
    center_coronal_axial_sagittal = [int((np.max(z) - np.min(z)) / 2 + np.min(z)),
                                     int((np.max(y) - np.min(y)) / 2 + np.min(y)),
                                     int((np.max(x) - np.min(x)) / 2) + np.min(x)]
    return center_coronal_axial_sagittal


def getCropBuff_byCenter(buff_zyx, center_sagittal_axial_coronal, expect_size_sagittal_axial_coronal, file_name=None,
                         record_file=None):
    ## init
    expect_size_S, expect_size_A, expect_size_C = expect_size_sagittal_axial_coronal
    iS, iH, iW = buff_zyx.shape
    cropbuff_sagittal_axial_coronal = np.zeros((expect_size_S, expect_size_A, expect_size_C), dtype=buff_zyx.dtype)
    c_z_sagittal, c_y_axial, c_x_coronal = center_sagittal_axial_coronal
    ## real crop range
    half_z_1 = expect_size_S // 2
    half_z_2 = expect_size_S - half_z_1
    half_y_1 = expect_size_A // 2
    half_y_2 = expect_size_A - half_y_1
    half_x_1 = expect_size_C // 2
    half_x_2 = expect_size_C - half_x_1
    ## z, sagittal
    low_z = c_z_sagittal - half_z_1 + 1
    if low_z < 0:
        print("%s has low sagittal!!!" % (file_name))
        if record_file != None:
            record_file.write("%s has low sagittal!!!\n" % (file_name))
        local_l_z = -low_z
        low_z = 0
    else:
        local_l_z = 0
    up_z = c_z_sagittal + half_z_2 + 1
    if up_z >= iS:
        print("%s has up sagittal!!!" % (file_name))
        if record_file != None:
            record_file.write("%s has up sagittal!!!\n" % (file_name))
        local_u_z = expect_size_S - (up_z - iS)  # (up_z - iS + 1)
        up_z = iS
    else:
        local_u_z = expect_size_S
    ## y, axial
    low_y = c_y_axial - half_y_1 + 1
    if low_y < 0:
        print("%s has low axial!!!" % (file_name))
        if record_file != None:
            record_file.write("%s has low axial!!!\n" % (file_name))
        local_l_y = -low_y
        low_y = 0
    else:
        local_l_y = 0
    up_y = c_y_axial + half_y_2 + 1
    if up_y >= iH:
        print("%s has up axial!!!" % (file_name))
        if record_file != None:
            record_file.write("%s has up axial!!!\n" % (file_name))
        local_u_y = expect_size_A - (up_y - iH)  # (up_y - iH + 1)
        up_y = iH
    else:
        local_u_y = expect_size_A
    ## x, coronal
    low_x = c_x_coronal - half_x_1 + 1
    if low_x < 0:
        print("%s has low coronal!!!" % (file_name))
        if record_file != None:
            record_file.write("%s has low coronal!!!\n" % (file_name))
        local_l_x = -low_x
        low_x = 0
    else:
        local_l_x = 0
    up_x = c_x_coronal + half_x_2 + 1
    if up_x >= iW:
        print("%s has up coronal!!!" % (file_name))
        if record_file != None:
            record_file.write("%s has up coronal!!!\n" % (file_name))
        local_u_x = expect_size_C - (up_x - iW)  # (up_x - iW + 1)
        up_x = iW
    else:
        local_u_x = expect_size_C
    cropbuff_sagittal_axial_coronal[local_l_z:local_u_z, local_l_y:local_u_y, local_l_x:local_u_x] = \
        buff_zyx[low_z:up_z, low_y:up_y, low_x:up_x]
    ##
    return cropbuff_sagittal_axial_coronal


def convert_folder(in_folder, out_folder, isRight, truth_name="label"):
    new_spacing = [1.0, 1.0, 1.0]
    name = ''
    axis = 2
    cropsize = (80, 144, 144)
    image_file = get_image(in_folder, name)
    out_file = os.path.abspath(os.path.join(out_folder, "processed_original" + ".nii.gz"))
    out_label_file = os.path.abspath(os.path.join(out_folder, "truth.nii.gz"))
    try:
        truth_file = get_image(in_folder, truth_name)
    except RuntimeError:
        truth_file = get_image(in_folder, truth_name.split("_")[0])
    label_file = sitk.ReadImage(truth_file)
    base_image = sitk.ReadImage(image_file)
    new_img_sitk = resamplingImage_Sp2(base_image, sp2=new_spacing, sampling_way=sitk.sitkLinear,
                                       sampling_pixel_t=sitk.sitkInt16)
    label_buffer = sitk.GetArrayFromImage(label_file)
    # Regularize labels
    label_buffer[np.where(label_buffer == 3)] = 2
    # End regularize labels
    new_lb_sitk = convBufferToItkImage(label_buffer, label_file)
    # Resampling the label to spacing [1,1,1]
    new_lb_sitk = resamplingImage_Sp2(new_lb_sitk, sp2=new_spacing)
    # Cropping the label base on the center of the valid label area into 80,144,144
    truth_np = getCropBuff_byCenter(sitk.GetArrayFromImage(new_lb_sitk),
                                    getTargetCenter_fromBuffL(sitk.GetArrayFromImage(new_lb_sitk)), cropsize)
    # N4 Bias Field Correction
    # The try except statement will catch abnormality in the N4 Bias Field Correction integrated in the SimpleITK
    try:
        N4 = mriN4BiasCorrection_sitkImage(new_img_sitk, numOfIters=500)
        gc.collect()
    except:
        print("Error has occurred\n")
        print("Current State:\n")
        print("Spacing:", new_img_sitk.GetSpacing(), "\n")
        print("Origin:", new_img_sitk.GetOrigin(), "\n")
        N4 = mriN4BiasCorrection_sitkImage(new_img_sitk, numOfIters=500)
        gc.collect()
    N4_np = sitk.GetArrayFromImage(N4)
    # Make all numbers in the array to the range [0,255)
    N4_positived = convBufferToItkImage(exposure.rescale_intensity(sitk.GetArrayFromImage(N4), out_range=(0, 255)), N4)
    N4_pos_np = sitk.GetArrayFromImage(N4_positived)
    # Crop Image centered on valid label area into (80,144,144)
    cropped_np = getCropBuff_byCenter(N4_pos_np, getTargetCenter_fromBuffL(sitk.GetArrayFromImage(new_lb_sitk)),
                                      cropsize)
    # Flip the right shoulder image to left
    if isRight:
        cropped_np = np.flip(cropped_np, axis)
    cropped = sitk.GetImageFromArray(cropped_np)
    cropped.SetDirection(N4_positived.GetDirection())
    cropped.SetSpacing(N4_positived.GetSpacing())
    # Flip the right shoulder label to left
    if isRight:
        truth_np = np.flip(truth_np, axis)
    truth_processed = sitk.GetImageFromArray(truth_np)
    truth_processed.SetDirection(new_lb_sitk.GetDirection())
    truth_processed.SetSpacing(new_lb_sitk.GetSpacing())
    # Write To File
    sitk.WriteImage(truth_processed, out_label_file)
    sitk.WriteImage(cropped, out_file)


# For enumerating the image folder in the given source


def preprocess(in_folder, out_folder, isRight=False):
    overwrite = True
    for subject_folder in glob.glob(os.path.join(in_folder, "*", "*")):
        if os.path.isdir(subject_folder):
            subject = os.path.basename(subject_folder)
            new_subject_folder = os.path.join(out_folder,
                                              subject)
            if not os.path.exists(new_subject_folder) or overwrite:
                if not os.path.exists(new_subject_folder):
                    os.makedirs(new_subject_folder)
                convert_folder(subject_folder, new_subject_folder, isRight)


# For augmenting and normalizing the data files


def augment(in_folder_augment, out_folder_augment, template):
    for subject_folder in glob.glob(os.path.join(in_folder_augment, "*", "*")):
        if os.path.isdir(subject_folder):
            subject = os.path.basename(subject_folder)
            new_subject_folder = os.path.join(out_folder_augment, os.path.basename(os.path.dirname(subject_folder)),
                                              subject)
            if not os.path.exists(new_subject_folder):
                os.makedirs(new_subject_folder)
            curr_img = sitk.ReadImage(os.path.join(subject_folder, "processed_original.nii.gz"))
            curr_np = sitk.GetArrayFromImage(curr_img)
            curr_lb = sitk.ReadImage(os.path.join(subject_folder, "truth.nii.gz"))
            sitk.WriteImage(curr_lb, os.path.join(new_subject_folder, "truth.nii.gz"))
            for i in range(0, 6):
                mapped_img_np = hist_match(curr_np, template[i])
                mapped_img = sitk.GetImageFromArray(mapped_img_np)
                mapped_img.SetDirection(curr_img.GetDirection())
                mapped_img.SetSpacing(curr_img.GetSpacing())
                sitk.WriteImage(mapped_img, os.path.join(new_subject_folder, "Augmented_" + str(i) + ".nii.gz"))


def main():
    # We encounter some issue in the SimpleITK library and we tried to enforce the garbage collector to resolve that
    # issue.
    gc.enable()
    preprocess("./Raw-Data/Left", "./Processed-Data/Left", isRight=False)
    preprocess("./Raw-Data/Right", "./Processed-Data/Right", isRight=True)
    template = list()
    # These are templates for the histogram mapping
    template.append(sitk.GetArrayFromImage(sitk.ReadImage("./Processed-Data/Left/2449334/processed_original.nii.gz")))
    template.append(sitk.GetArrayFromImage(sitk.ReadImage("./Processed-Data/Left/2360402R/processed_original.nii.gz")))
    template.append(sitk.GetArrayFromImage(sitk.ReadImage("./Processed-Data/Left/696891/processed_original.nii.gz")))
    template.append(sitk.GetArrayFromImage(sitk.ReadImage("./Processed-Data/Left/5014490R/processed_original.nii.gz")))
    template.append(sitk.GetArrayFromImage(sitk.ReadImage("./Processed-Data/Left/5003078R/processed_original.nii.gz")))
    template.append(sitk.GetArrayFromImage(sitk.ReadImage("./Processed-Data/Left/900145104/processed_original.nii.gz")))
    augment("./Processed-Data/", "../Augmenting/Augmented/", template)


if __name__ == "__main__":
    main()
