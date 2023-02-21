"""
This module contains the functions to estimate the noise model of the camera.

Based on the methods described in [1]_.
[1]_ Huang, F., Hartwich, T., Rivera-Molina, F. et al.
Video-rate nanoscopy using sCMOS camera–specific single-molecule localization
algorithms.
Nat Methods 10, 653–658 (2013). https://doi.org/10.1038/nmeth.2488
"""

from pathlib import Path
from typing import List, Tuple, Union

import numpy as np
import numpy.typing as npt
import tifffile as tif
import vodex as vx
from tqdm import tqdm


class sCMOSNoise:
    """A noise model for sCMOS cameras that assumes a fixed noise level
    as described in [1]_.

    Args:
        offset : The offset of the camera. Shape: (x, y)
        variance : The variance of the camera. Shape: (x, y)
        gain : The gain of the camera. Shape: (x, y)

    Attributes:
        offset : The offset of the camera. Shape: (x, y)
        variance : The variance of the camera. Shape: (x, y)
        gain : The gain of the camera. Shape: (x, y)

    """

    def __init__(
        self,
        offset: npt.NDArray[float],
        variance: npt.NDArray[float],
        gain: npt.NDArray[float],
    ):
        assert offset.shape == variance.shape == gain.shape, (
            f"Offset, variance, and gain must have the same shape. "
            f"Got {offset.shape}, {variance.shape}, and {gain.shape}."
        )
        self.offset: npt.NDArray[float] = offset
        self.variance: npt.NDArray[float] = variance
        self.gain: npt.NDArray[float] = gain

    @staticmethod
    def _two_sample_mean_and_var_from_stats(
        x: Tuple[int, npt.NDArray[float], npt.NDArray[float]],
        y: Tuple[int, npt.NDArray[float], npt.NDArray[float]] = None,
    ) -> npt.NDArray[int]:
        """
        Calculates the mean and variance of two samples combined, given their
        individual means and variances. If only one sample is given,
        the mean and variance of that sample is returned.

        Total variance of two samples,
        one of size n, sample mean <x>, and sample variance Sx^2,
        and one of size m, sample mean <y>, and sample variance Sy^2,
        is given by
        average = n*x + m*y / (n + m)
        var = ( (n-1)*Sx^2 + (m-1)*Sy^2 + n*m*(x-y)^2 / (n+m) ) / (n+m-1)
        (and assuming individual variances are calculated with
        Bessel's correction: n -1 instead of n in the denominator):

        Args:
            x : The first sample statistics: (n, <x>, Sx^2),
                where n is the number of samples, <x> is the mean (shape: x,y),
                and Sx is the standard deviation (shape: x,y).
            y : The second sample statistics: (m, <y>, Sy^2),
                where m is the number of samples, <y> is the mean (shape: x,y),
                and Sy is the standard deviation (shape: x,y).

        Returns:
            num_samples (int): The total number of samples combined.
            avg (npt.NDArray[float]): The mean of the two samples combined.
                Shape: (x, y)
            std (npt.NDArray[float]): The standard deviation of the two samples
                combined. Shape: (x, y)
        """
        if y is None:
            num_samples, avg, var = x
        else:
            n, x_mean, x_var = x
            m, y_mean, y_var = y
            num_samples = n + m
            avg = (n * x_mean + m * y_mean) / (n + m)
            var = (
                (n - 1) * x_var
                + (m - 1) * y_var
                + (n * m * (x_mean - y_mean) ** 2) / (n + m)
            ) / (n + m - 1)

        return num_samples, avg, var

    @staticmethod
    def _batch_mean_and_var_from_files(
        frames_dir: Union[str, Path],
        batch_size: int = None,
        file_type: str = "TIFF",
        verbose: bool = True,
    ) -> Tuple[npt.NDArray[float], npt.NDArray[float]]:
        """
        Calculates the mean and variance of the frames in batches.
        Based on the formula for the variance of two groups.

        Args:
            frames_dir : The directory with the frames to be processed.
                Shape: (n_frames, x, y)
            batch_size : The size of the batches,
                number of frames averaged at once.
                If None, all frames are averaged at once.

        Returns:
            average : The average of the frames per pixel. Shape: (x, y)
            var : The variance of the frames per pixel. Shape: (x, y)
        """
        # create a dummy experiment to load frames
        exp = vx.Experiment.create(
            vx.VolumeManager.from_dir(frames_dir, 1, file_type=file_type), []
        )
        n_frames = exp.db.get_n_frames()
        if batch_size is None:
            batch_size = n_frames
        n_batches = np.ceil(n_frames / batch_size).astype(int)

        if verbose:
            print(
                f"Calculating mean and standard deviation of {n_frames} "
                f"frames in {frames_dir}"
            )

        # load frames in batches and calculate mean and variance
        previous_batch_stats = None
        for i in range(n_batches):
            from_frame, to_frame = i * batch_size, min(
                (i + 1) * batch_size, n_frames
            )
            frame_ids = list(np.arange(from_frame, to_frame))
            frames = exp.load_volumes(frame_ids)

            n_batch = frames.shape[0]
            batch_mean = np.squeeze(np.mean(frames, axis=0))
            batch_var = np.squeeze(np.var(frames, axis=0, ddof=1))
            (
                num_samples,
                avg,
                var,
            ) = sCMOSNoise._two_sample_mean_and_var_from_stats(
                (n_batch, batch_mean, batch_var), previous_batch_stats
            )
            previous_batch_stats = (num_samples, avg, var)

        return avg, var

    @staticmethod
    def _offset_and_variance_from_files(
        zero_photon_frames_dir: Union[str, Path],
        batch_size=None,
        file_type: str = "TIFF",
        verbose: bool = True,
    ) -> Tuple[npt.NDArray[float], npt.NDArray[float]]:
        """Calculates the offset and variance of the camera.

        Args:
            zero_photon_frames_dir : The directory with the frames recorded
                with zero photons reaching the camera.
            batch_size : The number of frames to load at once. If None,
                all frames are loaded at once.

        Returns:
            offset : The offset of the camera. Shape: (x, y)
            variance : The variance of the camera. Shape: (x, y)
        """
        if not Path(zero_photon_frames_dir).is_dir():
            raise ValueError(
                f"The directory {zero_photon_frames_dir} does not exist."
            )

        return sCMOSNoise._batch_mean_and_var_from_files(
            zero_photon_frames_dir, batch_size, file_type, verbose
        )

    @staticmethod
    def _gain_from_stats(
        offset_g: npt.NDArray[float],
        offset: npt.NDArray[float],
        variance_g: npt.NDArray[float],
        variance: npt.NDArray[float],
    ) -> npt.NDArray[float]:
        """Calculates the gain of the camera from the statistics of the frames.

        Args:

            offset_g : The offset of the camera for different illumination
                levels for each pixel.
                Shape: (n_illumination_levels, x, y)
            offset : The offset of the camera for each pixel. Shape: (x, y)
            variance_g : The variance of the camera for different illumination
                levels for each pixel.
                Shape: (n_illumination_levels, x, y)
            variance : The variance of the camera for each pixel. Shape: (x, y)

        Returns:
            gain : The gain of the camera. Shape: (x, y)
        """
        n_rows, n_cols = offset.shape

        # definition of A and B , see doi:10.1038/nmeth.2488
        # Section 2.3, Eq. (2.4)
        A = variance_g - variance
        B = offset_g - offset

        # # calculate gain
        # gain = np.invert(B@B.T)@B@A.T

        gain = np.zeros((n_rows, n_cols))
        for i in range(n_rows):
            for j in range(n_cols):
                # must be a column vector:
                Aij = A[:, i, j][:, np.newaxis]
                Bij = B[:, i, j][:, np.newaxis]
                # from Eq. 2.5 in supplementary note from [1]_ : BiT* gi = AiT
                gain[i, j] = (np.linalg.lstsq(Bij, Aij, rcond=None))[0]

        if np.any(gain <= 0):
            raise ValueError("Some gain values are <= 0.")
        return gain

    @staticmethod
    def _gain_from_frames(
        illuminated_frames: npt.NDArray[int],
        offset: npt.NDArray[float],
        variance: npt.NDArray[float],
    ) -> npt.NDArray[float]:
        """Calculates the gain of the camera.

        Args:
            illuminated_frames : The frames recorded with different number of
                photons reaching the camera.
                Shape: (n_illumination_levels, n_frames, x, y)
            offset : The offset of the camera. Shape: (x, y)
            variance : The variance of the camera. Shape: (x, y)

        Returns:
            gain : The gain of the camera.
                Shape: (x, y)
        """

        # mean illumination per level (in ADU) and variance
        # is used to calculate the gain,
        # For details, see doi:10.1038/nmeth.2488 Section 2.3

        offset_g = np.squeeze(np.mean(illuminated_frames, axis=1))
        variance_g = np.squeeze(np.var(illuminated_frames, axis=1))

        return sCMOSNoise._gain_from_stats(
            offset_g, offset, variance_g, variance
        )

    @staticmethod
    def _gain_from_files(
        illuminated_frames_dirs: List[Union[str, Path]],
        offset,
        variance,
        batch_size=None,
        file_type: str = "TIFF",
        verbose: bool = True,
    ) -> npt.NDArray[float]:
        """
        Calculates the gain of the camera. mean illumination per level (in ADU)
        and variance is used to calculate the gain,
        For details, see doi:10.1038/nmeth.2488 Section 2.3

        Args:
            illuminated_frames_dirs : The directory with the frames recorded
                with different number of photons reaching the camera.
                Shape: (n_illumination_levels, n_frames, x, y)
            offset : The offset of the camera. Shape: (x, y)
            variance : The variance of the camera. Shape: (x, y)
            batch_size : The number of frames to load at once.
                If None, all frames are loaded at once.

        Returns:
            gain : The gain of the camera.
                Shape: (x, y)
        """
        # check that all folders exist first before loading any frames
        for il_dir in illuminated_frames_dirs:
            if not Path(il_dir).is_dir():
                raise ValueError(f"The directory {il_dir} does not exist.")

        n_levels = len(illuminated_frames_dirs)
        offset_g = np.zeros((n_levels, *offset.shape))
        variance_g = np.zeros((n_levels, *variance.shape))

        for i_il_dir, il_dir in enumerate(illuminated_frames_dirs):
            (
                offset_g[i_il_dir],
                variance_g[i_il_dir],
            ) = sCMOSNoise._batch_mean_and_var_from_files(
                il_dir, batch_size, file_type, verbose
            )

        return sCMOSNoise._gain_from_stats(
            offset_g, offset, variance_g, variance
        )

    @classmethod
    def _from_calibration_frames(
        cls,
        zero_photon_frames: npt.NDArray[int],
        illuminated_frames: npt.NDArray[int],
    ) -> "sCMOSNoise":
        """
        Creates a noise model from calibration frames. Probably not usable
        for any real-life scenario due to the large RAM requirement.

        Args:
            zero_photon_frames : The frames recorded with zero photons reaching
                the camera.
                Shape: (n_frames, x, y)
                (cover the camera with a cap or foil).
            illuminated_frames : The frames recorded with different number of
                photons reaching the camera.
                Shape: (n_illumination_levels, n_frames, x, y)
        """

        offset: npt.NDArray[int] = np.mean(zero_photon_frames, axis=0)
        variance: npt.NDArray[int] = np.var(zero_photon_frames, axis=0, ddof=1)
        gain: npt.NDArray[float] = cls._gain_from_frames(
            illuminated_frames, offset, variance
        )
        return cls(offset, variance, gain)

    @classmethod
    def _from_calibration_files(
        cls,
        zero_photon_folder: Union[str, Path],
        illuminated_frames_folders: List[Union[str, Path]],
        batch_size: int = None,
        file_type="TIFF",
        verbose=True,
    ) -> "sCMOSNoise":
        """Creates a noise model from calibration files.
        Loads and processes the frames from the folders sequentially,
            which could be better if the files are large.

        Args:
            zero_photon_folder : The folder containing the frames recorded with
                zero photons reaching the camera.
            illuminated_frames_folders : The folders containing the frames
                recorded with different number of photons
                reaching the camera.
            batch_size : The number of frames to load at once.
                If None, all frames are loaded at once.
            file_type : The file type of the calibration files.
                Default: 'TIFF' verbose : If True,
                prints the progress of the function. Default: True
            verbose : If True, prints the progress of the function.
                Default: True

        Returns:
            noise_model : The noise model.
        """
        # Process zero photon frames
        offset, variance = cls._offset_and_variance_from_files(
            zero_photon_folder,
            batch_size=batch_size,
            file_type=file_type,
            verbose=verbose,
        )
        gain = cls._gain_from_files(
            illuminated_frames_folders,
            offset,
            variance,
            batch_size=batch_size,
            file_type=file_type,
            verbose=verbose,
        )

        return cls(offset, variance, gain)

    @classmethod
    def estimate(
        cls,
        zero_photon_data: Union[npt.NDArray[int], Union[str, Path]],
        illuminated_data: Union[npt.NDArray[int], List[Union[str, Path]]],
        batch_size: int = None,
        file_type="TIFF",
        verbose=True,
    ) -> "sCMOSNoise":
        """
        Estimates the noise model from calibration frames or files.
        zero_photon_data and illuminated_data must be either both numpy arrays
        or both folders with the calibration data.

        Args:
            zero_photon_data : The frames recorded with zero photons reaching
                the camera as a numpy array or the folder containing the frames
                recorded with zero photons reaching the camera.
                Shape: (n_frames, x, y)
                (cover the camera with a cap or foil).
            illuminated_data : The frames recorded with different number of
                photons reaching the camera or the folders containing the
                frames recorded with different number of photons
                reaching the camera.
                Shape: (n_illumination_levels, n_frames, x, y)
            batch_size : The number of frames to load at once. If None,
                all frames are loaded at once.
            file_type : The file type of the calibration files.
                Default: 'TIFF'
            verbose : If True, prints the progress of the function.
                Default: True

        Returns:
            noise_model : The noise model.
        """
        if isinstance(zero_photon_data, np.ndarray) and isinstance(
            illuminated_data, np.ndarray
        ):
            return cls._from_calibration_frames(
                zero_photon_data, illuminated_data
            )
        elif isinstance(zero_photon_data, (str, Path)) and isinstance(
            illuminated_data, List
        ):
            return cls._from_calibration_files(
                zero_photon_data,
                illuminated_data,
                batch_size,
                file_type,
                verbose,
            )
        else:
            raise ValueError(
                "Either zero_photon_data and illuminated_data must be either "
                "both numpy arrays or both be folders containing the "
                "calibration files."
            )

    def save(self, folder_path: str, verbose: bool = True) -> None:
        """Saves the noise model to a file.

        Args:
            folder_path : The path to the folder to save files in.
        """
        tif.imwrite(
            Path(folder_path, "calibration_offset.tif"),
            data=self.offset.astype(np.float32),
            imagej=True,
        )
        tif.imwrite(
            Path(folder_path, "calibration_variance.tif"),
            data=self.variance.astype(np.float32),
            imagej=True,
        )
        tif.imwrite(
            Path(folder_path, "calibration_gain.tif"),
            data=self.gain.astype(np.float32),
            imagej=True,
        )

        if verbose:
            print(
                f"Calibration files saved to {folder_path}. \n"
                f"Offset file: calibration_offset.tif; \n"
                f"Variance file: calibration_variance.tif; \n"
                f"Gain file: calibration_gain.tif."
            )

    @classmethod
    def load(cls, folder_path: str) -> "sCMOSNoise":
        """Loads the noise model from a file.

        Args:
            folder_path : The path to the folder with the .

        Returns:
            noise_model : The noise model.
        """
        offset = tif.imread(
            Path(folder_path, "calibration_offset.tif").as_posix()
        )
        variance = tif.imread(
            Path(folder_path, "calibration_variance.tif").as_posix()
        )
        gain = tif.imread(Path(folder_path, "calibration_gain.tif").as_posix())
        return cls(offset, variance, gain)

    def denoise_frames(
        self,
        frames: npt.NDArray[int],
        clip=True,
        value_range: Tuple[Union[int, float], Union[int, float]] = None,
    ) -> npt.NDArray[int]:
        """Denoises an image using the estimated noise model.

        Args:
            frames : The frames to be denoised. Shape: (n_frames, x, y)
            clip : If True, values outside the value_range will be clipped
            after denoising.
            value_range : The value range of the denoised_frames.
                Any values outside this range will be clipped. If None,
                the value range is set to
                (np.min(frames), np.percentile(denoised_frames, 99).
                Default: None

        Returns:
            denoised_frames : The denoised image.

        """
        # remove the offset and divide by the gain

        denoised_frames = (frames - self.offset) / self.gain
        # clip the values if necessary
        if clip:
            if value_range is None:
                value_range = (
                    np.min(denoised_frames),
                    np.percentile(denoised_frames, 99),
                )
            if value_range[0] > value_range[1]:
                raise ValueError(
                    "The value range must be a tuple of (min, max)."
                )
            if np.any(denoised_frames <= value_range[0]):
                denoised_frames[denoised_frames <= value_range[0]] = (
                    value_range[0] + 1e-6
                )
            if np.any(denoised_frames >= value_range[1]):
                denoised_frames[denoised_frames > value_range[1]] = (
                    value_range[1] - 1e-6
                )

        return denoised_frames

    def denoise_files(
        self,
        image_path: Union[str, Path],
        save_path: Union[str, Path],
        batch_size: int = None,
        file_type: str = "TIFF",
        scale: float = 1,
        clip: bool = True,
        value_range: Tuple[Union[int, float], Union[int, float]] = None,
        verbose: bool = True,
    ) -> None:
        """Denoises the files in a folder using the estimated noise model.

        Args:
            image_path : The path to the folder with the files to be denoised.
            save_path : The path to the folder to save the denoised files in.
            batch_size : The number of frames to load at once.
                If None, all frames are loaded at once.
            file_type : The file type of the files to be denoised.
                Default: 'TIFF'
            scale : The scale of the images. Default: 1
            clip : If True, values outside the value_range will be clipped
                after denoising.
            value_range : The value range of the denoised_frames.
                Any values outside this range will be clipped. If None,
                the value range is set to
                (np.min(frames), np.percentile(frames, 99). Default: None
            verbose : If True, prints the progress of the function.
                Default: True
        """
        # create a dummy experiment to load frames
        exp = vx.Experiment.create(
            vx.VolumeManager.from_dir(image_path, 1, file_type=file_type), []
        )
        n_frames = exp.db.get_n_frames()

        if batch_size is None:
            batch_size = n_frames
        n_batches = np.ceil(n_frames / batch_size).astype(int)

        if verbose:
            print(
                f"Denoising {n_frames} frames in {image_path} in "
                f"{n_batches} batches of {batch_size} frames."
            )

        # load frames in batches, denoise and save them
        for i in tqdm(range(n_batches), disable=not verbose):
            from_frame, to_frame = i * batch_size, min(
                (i + 1) * batch_size, n_frames
            )
            frame_ids = list(np.arange(from_frame, to_frame))
            frames = exp.load_volumes(frame_ids)
            denoised_frames = (
                self.denoise_frames(frames, clip, value_range) * scale
            )
            tif.imwrite(
                Path(
                    save_path,
                    f"denoised_{str(i).zfill(len(str(n_batches)))}.tif",
                ),
                data=denoised_frames.astype(exp.loader.loader.data_type),
                imagej=True,
            )
