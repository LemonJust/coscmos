"""
This file is used to create the noise model for the sCMOS camera used in the
numerosity 2023 experiments for the 1hz dataset fish code is hz## .
"""

import coscmos as cm
from pathlib import Path
import vodex as vx
import warnings

callibration_folder = 'D:/Data/Numerosity/denoising_2023/'
zero_photon_folder = Path(callibration_folder,
                          '20230520_darkcurrentnoise_14msexp_fullbrainROI_hotpixoff_30kframes_1')
illuminated_frames_folders = [
    Path(callibration_folder,
         '20230520_8photon_14msexp_fullbrainROI_hotpixoff_10kframes_1'),
    Path(callibration_folder,
         '20230520_16photon_14msexp_fullbrainROI_hotpixoff_10kframes_1'),
    Path(callibration_folder,
         '20230520_32photon_14msexp_fullbrainROI_hotpixoff_10kframes_1'),
    Path(callibration_folder,
         '20230520_64photon_14msexp_fullbrainROI_hotpixoff_10kframes_1'),
    Path(callibration_folder,
         '20230520_128photon_14msexp_fullbrainROI_hotpixoff_10kframes_1'),
    Path(callibration_folder,
         '20230520_256photon_14msexp_fullbrainROI_hotpixoff_10kframes_1'),
    Path(callibration_folder,
         '20230520_512photon_14msexp_fullbrainROI_hotpixoff_10kframes_1'),
    Path(callibration_folder,
         '20230520_1024photon_14msexp_fullbrainROI_hotpixoff_10kframes_1')
    # ,
    # Path(callibration_folder,
    #      '20230520_1600photon_14msexp_fullbrainROI_hotpixoff_10kframes_1'),
    # Path(callibration_folder,
    #      '20230520_2048photon_14msexp_fullbrainROI_hotpixoff_10kframes_1')
]


def check_n_frames(zero_photon_folder, illuminated_frames_folders,
                   n_frames_zero_photon=30000, n_frames_illuminated=10000):
    def check_folder(folder, expected_n_frames, to_stop):
        # create dummy experiment to get the number of frames
        dummy_experiment = vx.Experiment.create(
            vx.VolumeManager.from_dir(folder, 1), []
        )
        n_frames = dummy_experiment.db.get_n_frames()

        print(f'Number of frames in {folder}: {n_frames}')

        if n_frames != expected_n_frames:
            to_stop = 1
            warnings.warn(
                f'Number of frames in {folder} '
                f'is not {expected_n_frames}')
        return to_stop

    # Check the number of frames in zero_photon_folder
    to_stop = check_folder(zero_photon_folder, n_frames_zero_photon, 0)
    # Check the number of frames in illuminated_frames_folders
    for folder in illuminated_frames_folders:
        to_stop = check_folder(folder, n_frames_illuminated, to_stop)

    if to_stop:
        raise ValueError('Number of frames in some folders is not correct')


if __name__ == "__main__":
    check_n_frames(zero_photon_folder, illuminated_frames_folders)
    # Create a noise

    noise_model = cm.sCMOSNoise.estimate(zero_photon_folder,
                                         illuminated_frames_folders,
                                         # scale=4,
                                         batch_size=2000)
    noise_model.save(
        'D:/Data/Numerosity/denoising_2023')
