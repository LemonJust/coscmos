from pathlib import Path

import numpy as np
import pytest
import tifffile as tif

# create calibration frames
# TODO: create more realistic calibration files using the realistic noise model
add_constant = 100
FRAME_SHAPE = (5, 3)
# 3000 frames with 5x3 pixels
n_z, mu_z, var_z = 3000, 0, 1
ZERO_PHOTON_FRAMES = np.random.normal(mu_z, var_z, (n_z, 5, 3)) + add_constant
# 2 illumination levels with 1000 frames each with 5x3 pixels
# this frames should result in offset_g ~ 1 and 2 and variance_g ~ 3 and 5
n_g, mu_g, var_g = 1000, (1, 2), (3, 5)
ILLUMINATED_FRAMES = (
    np.concatenate(
        [
            np.random.normal(mu_g[0], np.sqrt(var_g[0]), (1, n_g, 5, 3)),
            np.random.normal(mu_g[1], np.sqrt(var_g[1]), (1, n_g, 5, 3)),
        ]
    )
    + add_constant
)

# offset and variance from the generated samples
OFFSET = np.mean(ZERO_PHOTON_FRAMES, axis=0)
VARIANCE = np.var(ZERO_PHOTON_FRAMES, axis=0, ddof=1)

# offset and variance of the true distribution
TRUE_OFFSET = np.zeros((5, 3)) + add_constant
TRUE_VARIANCE = np.ones((5, 3))

# calculate expected gain for each pixel:
a = [(var_g[0] - var_z), (var_g[1] - var_z)]
b = [(mu_g[0] - mu_z), (mu_g[1] - mu_z)]
TRUE_GAIN = (
    np.ones((5, 3)) * (a[0] * b[0] + a[1] * b[1]) / (b[0] ** 2 + b[1] ** 2)
)

# create images to be denoised
IMAGES = np.ones((10, 5, 3))
NOISY_IMAGES = IMAGES * TRUE_GAIN + TRUE_OFFSET


# create calibration files: before tests run, will create a temporary
# directory with the calibration files the folder will be deleted after the
# tests are finished
@pytest.fixture(autouse=True, scope="session")
def calibration_files(tmpdir_factory):
    # create the folders
    datadir = tmpdir_factory.mktemp("tmp")
    zero_photon_folder = Path(datadir, "zero_photon")
    n_levels = 2
    illuminated_frames_folders = [
        Path(datadir, f"illumination_level_{i}") for i in range(n_levels)
    ]

    # add files into the folders
    zero_photon_folder.mkdir()
    # 10 files with 300 frames each
    for i in range(10):
        from_ = i * int(n_z / 10)
        to_ = (i + 1) * int(n_z / 10)
        tif.imwrite(
            Path(zero_photon_folder, f"illumination_{i}.tif"),
            data=ZERO_PHOTON_FRAMES[from_:to_, :, :].astype(np.float32),
            shape=(300, 5, 3),
            metadata={"axes": "TYX"},
            imagej=True,
        )

    for i_folder, folder in enumerate(illuminated_frames_folders):
        folder.mkdir()
        # 10 files with 100 frames each
        for i in range(10):
            from_ = i * int(n_g / 10)
            to_ = (i + 1) * int(n_g / 10)
            tif.imwrite(
                Path(folder, f"illumination_{i}.tif"),
                data=ILLUMINATED_FRAMES[i_folder, from_:to_, :, :].astype(
                    np.float32
                ),
                shape=(100, 5, 3),
                metadata={"axes": "TYX"},
                imagej=True,
            )

    return zero_photon_folder, illuminated_frames_folders


@pytest.fixture(autouse=True, scope="session")
def image_folder(tmpdir_factory):
    # create the folders
    image_folder = tmpdir_factory.mktemp("image_folder")

    tif.imwrite(
        Path(image_folder, "noisy_images.tif"),
        data=NOISY_IMAGES.astype(np.float32),
        shape=(10, 5, 3),
        metadata={"axes": "TYX"},
        imagej=True,
    )
    return image_folder
