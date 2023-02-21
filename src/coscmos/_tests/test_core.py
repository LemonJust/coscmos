from pathlib import Path

import numpy as np
import pytest
import tifffile as tif

from coscmos import sCMOSNoise

from .conftest import (
    FRAME_SHAPE,
    ILLUMINATED_FRAMES,
    IMAGES,
    NOISY_IMAGES,
    OFFSET,
    TRUE_GAIN,
    TRUE_OFFSET,
    TRUE_VARIANCE,
    VARIANCE,
    ZERO_PHOTON_FRAMES,
)


# class to test sCMOSNoise class
class TestSCMOSNoise:
    def test_init(self):
        offset = np.ones((5, 5), dtype=int)
        variance = np.ones((5, 5), dtype=int)
        gain = np.ones((5, 5), dtype=float)
        scmos_noise = sCMOSNoise(offset, variance, gain)

        assert isinstance(scmos_noise, sCMOSNoise)
        assert (scmos_noise.offset == offset).all()
        assert (scmos_noise.variance == variance).all()
        assert (scmos_noise.gain == gain).all()

        with pytest.raises(AssertionError) as e:
            sCMOSNoise(offset, variance, np.ones((10, 10), dtype=float))
        assert str(e.value) == (
            "Offset, variance, and gain must have the same shape. "
            "Got (5, 5), (5, 5), and (10, 10)."
        )

    def test_two_sample_mean_and_var_from_stats(self):
        x = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
        y = np.array([11, 12, 13, 14, 15])
        # require ddof=1 to get unbiased estimator of variance
        x_mean, x_std = np.mean(x), np.var(x, ddof=1)
        y_mean, y_std = np.mean(y), np.var(y, ddof=1)
        n, x1_mean, x1_std = sCMOSNoise._two_sample_mean_and_var_from_stats(
            (10, x_mean, x_std)
        )
        assert n == 10
        assert x_mean == x1_mean
        assert x_std == x1_std

        n, xy_mean, xy_std = sCMOSNoise._two_sample_mean_and_var_from_stats(
            (10, x_mean, x_std), (5, y_mean, y_std)
        )

        assert n == 15
        assert xy_mean == np.mean(np.concatenate((x, y)))
        assert xy_std == np.var(np.concatenate((x, y)), ddof=1)

    def test_batch_mean_and_var_from_files(self, calibration_files):
        zero_photon_folder, _ = calibration_files
        # test with zero photon frames
        # with batch size
        mean, var = sCMOSNoise._batch_mean_and_var_from_files(
            zero_photon_folder, batch_size=5
        )

        assert mean.shape == FRAME_SHAPE
        assert var.shape == FRAME_SHAPE

        assert np.allclose(mean, OFFSET)
        assert np.allclose(var, VARIANCE)
        # without batch size
        mean, var = sCMOSNoise._batch_mean_and_var_from_files(
            zero_photon_folder
        )
        assert np.allclose(mean, OFFSET)
        assert np.allclose(var, VARIANCE)

    def test_offset_and_variance_from_files(self, calibration_files):
        zero_photon_folder, _ = calibration_files
        # test with zero photon frames
        offset, variance = sCMOSNoise._offset_and_variance_from_files(
            zero_photon_folder
        )

        assert offset.shape == FRAME_SHAPE
        assert variance.shape == FRAME_SHAPE

        assert np.allclose(offset, OFFSET)
        assert np.allclose(variance, VARIANCE)

        # I wonder if it can break on some random occasions due to random ?
        assert np.allclose(np.round(offset), TRUE_OFFSET)
        assert np.allclose(np.round(variance), TRUE_VARIANCE)

        with pytest.raises(ValueError) as e:
            sCMOSNoise._offset_and_variance_from_files("I_am_not_existing_dir")
        assert (
            str(e.value)
            == "The directory I_am_not_existing_dir does not exist."
        )

    def test_gain_from_stats(self):
        # super simple test for now:
        # assume only one illumination level,
        # and a very small image of 5x3 pixels
        # ( following eq. 2.3 in supplementary note from [1]_
        # and simplifying for one illumination level:
        #  gain = (variance_g - variance) / (offset_g - offset)
        offset = np.zeros((5, 3))  # 0
        variance = np.ones((5, 3))  # 1
        # this values should result in a gain of 4 for each pixel:
        offset_g = np.ones((1, 5, 3))  # 1
        variance_g = np.ones((1, 5, 3)) * 5.0  # 5
        gain = sCMOSNoise._gain_from_stats(
            offset_g, offset, variance_g, variance
        )
        assert np.allclose(gain, np.ones((5, 3), dtype=float) * 4.0)
        # these in 1.33333.... (4/3):
        gain = sCMOSNoise._gain_from_stats(
            offset_g * 3, offset, variance_g, variance
        )
        assert np.allclose(gain, np.ones((5, 3), dtype=float) * (4 / 3))

        # test with two illumination levels
        # this values should result in a gain of 2 for each pixel
        # (since [1,2]*[1,2].T * [1,2]*[2, 4].T = 2)
        offset_g = np.ones((2, 5, 3))  # 1
        offset_g[1, :, :] = 2  # 2
        variance_g = np.ones((2, 5, 3)) * 3.0  # 3
        variance_g[1, :, :] = 5.0  # 5
        gain = sCMOSNoise._gain_from_stats(
            offset_g, offset, variance_g, variance
        )
        assert np.allclose(gain, np.ones((5, 3), dtype=float) * 2.0)

    def test_gain_from_frames(self):
        # test with two illumination levels
        gain = sCMOSNoise._gain_from_frames(
            ILLUMINATED_FRAMES, TRUE_OFFSET, TRUE_VARIANCE
        )

        assert np.allclose(np.round(gain), TRUE_GAIN)

    def test_gain_from_files(self, calibration_files):
        _, illuminated_frames_folders = calibration_files
        gain = sCMOSNoise._gain_from_files(
            illuminated_frames_folders, TRUE_OFFSET, TRUE_VARIANCE
        )
        assert np.allclose(np.round(gain), TRUE_GAIN)

        with pytest.raises(ValueError) as e:
            sCMOSNoise._gain_from_files(
                ["I_am_not_existing_dir"], TRUE_OFFSET, TRUE_VARIANCE
            )
        assert (
            str(e.value)
            == "The directory I_am_not_existing_dir does not exist."
        )

    def test_from_calibration_frames(self):
        # test with two illumination levels
        scmos_noise = sCMOSNoise._from_calibration_frames(
            ZERO_PHOTON_FRAMES, ILLUMINATED_FRAMES
        )

        assert scmos_noise.offset.shape == FRAME_SHAPE
        assert scmos_noise.variance.shape == FRAME_SHAPE
        assert scmos_noise.gain.shape == FRAME_SHAPE

        assert np.allclose(scmos_noise.offset, OFFSET)
        assert np.allclose(scmos_noise.variance, VARIANCE)
        assert np.allclose(np.round(scmos_noise.gain), TRUE_GAIN)

    def test_from_calibration_files(self, calibration_files):
        zero_photon_folder, illuminated_frames_folders = calibration_files
        scmos_noise = sCMOSNoise._from_calibration_files(
            zero_photon_folder, illuminated_frames_folders
        )

        assert scmos_noise.offset.shape == FRAME_SHAPE
        assert scmos_noise.variance.shape == FRAME_SHAPE
        assert scmos_noise.gain.shape == FRAME_SHAPE

        assert np.allclose(scmos_noise.offset, OFFSET)
        assert np.allclose(scmos_noise.variance, VARIANCE)
        assert np.allclose(np.round(scmos_noise.gain), TRUE_GAIN)

    def test_estimate(self, calibration_files):
        zero_photon_folder, illuminated_frames_folders = calibration_files
        # test with folders
        scmos_noise = sCMOSNoise.estimate(
            zero_photon_folder, illuminated_frames_folders
        )

        assert scmos_noise.offset.shape == FRAME_SHAPE
        assert scmos_noise.variance.shape == FRAME_SHAPE
        assert scmos_noise.gain.shape == FRAME_SHAPE

        assert np.allclose(scmos_noise.offset, OFFSET)
        assert np.allclose(scmos_noise.variance, VARIANCE)
        assert np.allclose(np.round(scmos_noise.gain), TRUE_GAIN)

        # test with frames
        scmos_noise = sCMOSNoise.estimate(
            ZERO_PHOTON_FRAMES, ILLUMINATED_FRAMES
        )

        assert scmos_noise.offset.shape == FRAME_SHAPE
        assert scmos_noise.variance.shape == FRAME_SHAPE
        assert scmos_noise.gain.shape == FRAME_SHAPE

        assert np.allclose(scmos_noise.offset, OFFSET)
        assert np.allclose(scmos_noise.variance, VARIANCE)
        assert np.allclose(np.round(scmos_noise.gain), TRUE_GAIN)

        # test with frames and files ( should break)
        with pytest.raises(ValueError) as e:
            sCMOSNoise.estimate(ZERO_PHOTON_FRAMES, illuminated_frames_folders)
        assert (
            str(e.value)
            == "Either zero_photon_data and illuminated_data must be "
            "either both numpy arrays or "
            "both be folders containing the calibration files."
        )

    def test_save(self, tmpdir):
        scmos_noise = sCMOSNoise(OFFSET, VARIANCE, TRUE_GAIN)
        scmos_noise.save(tmpdir)
        assert Path(tmpdir, "calibration_offset.tif").is_file()
        assert Path(tmpdir, "calibration_variance.tif").is_file()
        assert Path(tmpdir, "calibration_gain.tif").is_file()

    def test_load(self, tmpdir):
        scmos_noise = sCMOSNoise(OFFSET, VARIANCE, TRUE_GAIN)
        scmos_noise.save(tmpdir)
        scmos_noise_loaded = sCMOSNoise.load(tmpdir)
        assert np.allclose(scmos_noise_loaded.offset, OFFSET)
        assert np.allclose(scmos_noise_loaded.variance, VARIANCE)
        assert np.allclose(scmos_noise_loaded.gain, TRUE_GAIN)

    def test_denoise_frames(self):
        scmos_noise = sCMOSNoise(OFFSET, VARIANCE, TRUE_GAIN)
        denoised_images = scmos_noise.denoise_frames(
            NOISY_IMAGES, clip=False
        )  # test with numpy array
        assert np.allclose(np.round(denoised_images), IMAGES)

        image_w_outlier = np.copy(NOISY_IMAGES)
        image_w_outlier[0, 0, 0] = 10000
        denoised_images = scmos_noise.denoise_frames(
            image_w_outlier, clip=True
        )
        assert np.allclose(np.round(denoised_images), IMAGES)

        denoised_images = scmos_noise.denoise_frames(
            NOISY_IMAGES, clip=True, value_range=(0, 0.5)
        )
        assert np.allclose(denoised_images, IMAGES * 0.5)

    def test_denoise_files(self, image_folder):
        scmos_noise = sCMOSNoise(OFFSET, VARIANCE, TRUE_GAIN)

        save_path = Path(image_folder, "denoised")
        save_path.mkdir()

        # try no batch size
        scmos_noise.denoise_files(
            image_folder, save_path, clip=False
        )  # test with files
        denoised_images = tif.imread(str(Path(save_path, "denoised_0.tif")))
        assert denoised_images.shape == (10, 5, 3)
        assert np.allclose(np.round(denoised_images), IMAGES)

        # try batch size of 1 to make sure the files are named correctly
        # ( with 0 padding)
        scmos_noise.denoise_files(
            image_folder, save_path, batch_size=1, clip=False
        )  # test with files
        denoised_images = np.concatenate(
            [
                tif.imread(str(Path(save_path, f"denoised_0{i}.tif")))[
                    np.newaxis, :
                ]
                for i in range(10)
            ]
        )
        assert denoised_images.shape == (10, 5, 3)
        assert np.allclose(np.round(denoised_images), IMAGES)


if __name__ == "__main__":
    # run pytest tests
    pytest.main()
