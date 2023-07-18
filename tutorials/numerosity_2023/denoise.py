import coscmos as cm
from pathlib import Path
import vodex as vx
import warnings

if __name__ == "__main__":
    # load the noise model from the files
    noise_model = cm.sCMOSNoise.load(
        'D:/Code/repos/coscmos/models/bin1x1_low_8_illumination_levels')

    # noisy data
    data_folder = Path(
        'F:/Data/sCMOS_calibration/images_to_denoise/'
        '20230217_1x1_dn01_h2bgcamp6sPTU_2cycle_sa18_1v2v3v4v5_2P_2')
    save_folder = Path(
        'F:/Data/sCMOS_calibration/denoised_images/'
        '20230217_1x1_dn01_h2bgcamp6sPTU_2cycle_sa18_1v2v3v4v5_2P_2_denoised/'
        'bin1x1_low_8_illumination_levels')
    save_folder.mkdir(exist_ok=False, parents=True)
    # denoise
    noise_model.denoise_files(data_folder, save_folder,
                              add_value=100,
                              scale=100,
                              batch_size=2000)
