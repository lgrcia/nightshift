import numpy as np


# Much faster than the previous one
def _coverage(times, period):
    if period == 0:
        return 1
    else:
        ph = (times + 0.5 * period) % period - (0.5 * period)
        sph_in = np.sort(ph)
        sph_out = sph_in[::-1]

        sph_in_diff = np.abs(np.diff(sph_in))
        sph_out_diff = np.abs(np.diff(sph_out))

        df = np.min(np.diff(times))

        spaces_in = np.sort(
            sph_in[np.hstack([*np.argwhere(sph_in_diff > 4 * df).T, len(sph_in) - 1])]
        )
        spaces_out = np.sort(
            sph_out[np.hstack([*np.argwhere(sph_out_diff > 4 * df).T, len(sph_in) - 1])]
        )

        return np.sum(spaces_in - spaces_out) / period


vectorize_coverage = np.vectorize(_coverage, signature="(n),()->()")


def coverage(times: list, periods: np.array):
    """Returns the phase coverage for given period(s)

    Parameters
    ----------
    times : list
        a list of observed times
    periods : np.array
        an array of periods

    Returns
    -------
    np.array
        coverage of each period
    """

    return vectorize_coverage(times, periods)
