import SimpleITK as sitk
import radiomics
from radiomics import featureextractor
import numpy as np
import pandas as pd

# This function computes the radiomics from a given iamge.
# code prepared and used in previous lab studies by Hossam ElRoweidy
def extract_all_radiomics_features(all_img, all_mask, all_labels=None, all_pid = None,
                                   storage_dfn = None, dataset_name = None, voxelspacing=None, params=None, manualnormalize=False):
    feats = pd.DataFrame()
    if storage_dfn is None or dataset_name is None:
        print('Please, provide location to store computed features')
        return
    for i in range(len(all_img)):
        dumi= np.expand_dims(all_img[i, :, :],2)
        my_img = sitk.GetImageFromArray(dumi)
        dumm=np.expand_dims(all_mask[i, :, :],2)
        my_msk= sitk.GetImageFromArray(dumm)
        fvector = runpyradiomicsonimage(my_img, my_msk, voxelspacing, params=params, manualnormalize=manualnormalize)
        feats = feats.append(fvector, ignore_index=True)
    if all_labels is not None:
        feats['slice_label'] = all_labels
    if all_pid is not None:
        feats['pat_id'] = all_pid

    feats.to_csv(storage_dfn[:-4] + 'csv')

def runpyradiomicsonimage(img, mask, voxelspacing, params=None, manualnormalize=False):
    if manualnormalize:
        maskedimg = np.multiply(img, mask)
        maskedvalues = img[np.where(mask == 1)]
        minimg = np.quantile(maskedvalues, 0.01)
        maximg = np.quantile(maskedvalues, 0.99)
        target_max_img = 255
        imagenorm = (maskedimg - minimg) * target_max_img / (maximg - minimg)
        imagenorm[np.where(imagenorm > target_max_img)] = target_max_img
        imagenorm[np.where(imagenorm < 0)] = 0
        img = imagenorm

    #prepare the image and mask into SimpleITK
    imagesitk = img
    VoxelSpacing = np.transpose(voxelspacing.astype('float'))
    imagesitk.SetSpacing(VoxelSpacing)
    imagesitk.SetOrigin(np.zeros(np.shape(VoxelSpacing)))

    masksitk = mask
    masksitk.SetSpacing(VoxelSpacing)
    masksitk.SetOrigin(np.zeros(np.shape(VoxelSpacing)))

    # Prepare the settings for the pyradiomics feature extractor
    settings = {}
    settings['normalize'] = False
    settings['interpolator'] = 'sitkBSpline'
    settings['verbose'] = True
    settings['force2D'] = True
    settings['removeOutliers'] = False
    # first order specific settings:
    settings[
        'voxelArrayShift'] = 0
    settings['distances'] = [1]
    settings['weightingNorm'] = 'no_weighting'
    settings['symmetricalGLCM'] = True

    # Here overwrite all the settings parameters with the possible input kwargs
    accepted_settings_args = {'normalize', 'binWidth', 'resampledPixelSpacing', 'interpolator', 'verbose', 'force2D',
                              'voxelArrayShift', 'distances', 'weightingNorm', 'symmetricalGLCM', 'removeOutliers',
                              'binCount'}
    for key in params:
        if key in accepted_settings_args:
            settings[key] = params[key]

    extractor = featureextractor.RadiomicsFeatureExtractor(**settings)
    extractor.disableAllFeatures()

    if 'featureclass' in params.keys():
        featureclass = params['featureclass']
        for feature in featureclass:
            extractor.enableFeatureClassByName(feature)
    else:
        extractor.enableAllFeatures()

    if 'imagetypes' in params.keys():
        imagetypes = params['imagetypes']
        for imagetype in imagetypes:
            if imagetype == 'LoG':
                extractor.enableImageTypeByName('LoG', enabled=True, customArgs={'sigma': [1, 2, 3]})
            else:
                extractor.enableImageTypeByName(imagetype)
    else:
        extractor.enableAllImageTypes()

    radiomics.setVerbosity(50)  # Only critical warnings
    features = extractor.execute(imagesitk, masksitk, label=1)

    return features
