from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from daofind_opt import Detection, daofind_like_detect, robust_center_noise, subtract_local_background


@dataclass
class Source:
    y: float
    x: float
    peak: float
    flux: float
    snr: float
    area: float = 0.0
    method: str = ""


def dao_like_sources(
    image: np.ndarray,
    sigma: float = 5.0,
    fwhm: float = 3.0,
    filtsize: int = 25,
    max_peaks: int = 10000,
    min_separation: float = 3.0,
    exclude_border: int = 8,
) -> list[Source]:
    detections = daofind_like_detect(
        image,
        sigma=float(sigma),
        fwhm=float(fwhm),
        background_mode="local_mean",
        filtsize=int(filtsize),
        max_peaks=int(max_peaks),
        min_separation=float(min_separation),
        exclude_border=int(exclude_border),
    )
    return [dao_detection_to_source(det, "daofind_like") for det in detections]


def dao_detection_to_source(det: Detection, method: str) -> Source:
    return Source(
        y=float(det.y),
        x=float(det.x),
        peak=float(det.peak),
        flux=float(det.flux),
        snr=float(det.snr),
        method=method,
    )


def daostarfinder_sources(
    image: np.ndarray,
    sigma: float = 5.0,
    fwhm: float = 3.0,
    filtsize: int = 25,
    max_peaks: int = 10000,
) -> list[Source]:
    from photutils.detection import DAOStarFinder

    residual = subtract_local_background(image, "local_mean", int(filtsize))
    _, noise = robust_center_noise(residual)
    finder = DAOStarFinder(
        threshold=float(sigma) * max(float(noise), 1e-6),
        fwhm=float(fwhm),
        sigma_radius=1.5,
        sharplo=0.2,
        sharphi=1.0,
        roundlo=-1.0,
        roundhi=1.0,
        exclude_border=True,
    )
    table = finder(residual)
    if table is None or len(table) == 0:
        return []
    table.sort("peak")
    sources: list[Source] = []
    for row in reversed(table[-int(max_peaks) :]):
        peak = float(row["peak"])
        sources.append(
            Source(
                y=float(row["ycentroid"]),
                x=float(row["xcentroid"]),
                peak=peak,
                flux=float(row["flux"]),
                snr=peak / max(float(noise), 1e-6),
                method="daostarfinder",
            )
        )
    return sources


def sextractor_sources(
    image: np.ndarray,
    sigma: float = 5.0,
    minarea: int = 3,
    deblend: bool = True,
    max_sources: int = 10000,
) -> list[Source]:
    import sep

    arr = np.asarray(image, dtype=np.float32)
    bkg = sep.Background(arr)
    residual = arr - bkg.back()
    objects = sep.extract(
        residual,
        float(sigma),
        err=bkg.globalrms,
        minarea=int(minarea),
        deblend_nthresh=32 if deblend else 1,
        deblend_cont=0.005 if deblend else 1.0,
    )
    sources: list[Source] = []
    for obj in objects[: int(max_sources)]:
        peak = float(obj["peak"])
        sources.append(
            Source(
                y=float(obj["y"]),
                x=float(obj["x"]),
                peak=peak,
                flux=float(obj["flux"]),
                snr=peak / max(float(bkg.globalrms), 1e-6),
                area=float(obj["npix"]),
                method="sextractor_sep",
            )
        )
    return sources
