"""Compare two fixed fragment predictions on held-out windows of the measured spectrum."""

import numpy as np

from fit_fragment_reliability import evidence


ARMS = ('cosine', 'heldout_template', 'heldout_with_common')
RIDGE = 0.05
WINDOWS = 4


def align(peaks):
    """Give every observed and predicted peak a common, order-independent bin."""
    masses = np.concatenate([m for m, _ in peaks])
    areas = np.concatenate([i for _, i in peaks])
    series = np.repeat(np.arange(len(peaks)), [len(m) for m, _ in peaks])
    order = np.argsort(masses, kind='stable')
    centres, bins = [], np.zeros((len(masses), len(peaks)))
    for index in order:
        mass = masses[index]
        if not centres or mass - centres[-1] > max(0.01, centres[-1] * 20e-6):
            centres.append(mass)
        bins[len(centres) - 1, series[index]] += areas[index]
    # Unmatched peaks have a zero in the corresponding column.
    return np.asarray(centres), np.sqrt(bins[:len(centres)])


def nonnegative_ridge(design, measured):
    """Solve the one- or two-column nonnegative ridge problem exactly."""
    width = design.shape[1]
    gram = design.T @ design + RIDGE * np.eye(width)
    rhs = design.T @ measured
    if width == 1:
        return np.maximum(0.0, rhs / np.diag(gram))
    if width != 2:
        raise ValueError('Expected one spectroscopic template and optional shared signal')
    axes = np.diag(np.maximum(0.0, rhs / np.diag(gram)))
    joint = np.linalg.solve(gram, rhs)
    choices = np.vstack((np.zeros(2), axes, joint)) if np.all(joint >= 0) else np.vstack((np.zeros(2), axes))
    values = np.einsum('ij,jk,ik->i', choices, gram, choices) - 2 * choices @ rhs
    return choices[np.argmin(values)]


def heldout_error(design, measured, window_ids):
    squared_error = 0.0
    for heldout in np.unique(window_ids):
        train = window_ids != heldout
        fit = nonnegative_ridge(design[train], measured[train])
        residual = measured[~train] - design[~train] @ fit
        squared_error += float(residual @ residual)
    return squared_error


def reconstruction(query, pair, precursor):
    centres, columns = align((query, *pair))
    measured, a, b = columns.T
    window_ids = np.clip((WINDOWS * centres / precursor).astype(np.intp), 0, WINDOWS - 1)
    common = np.minimum(a, b)
    return {
        'heldout_template': [1.0 - heldout_error(v[:, None], measured, window_ids) for v in (a, b)],
        'heldout_with_common': [1.0 - heldout_error(np.column_stack((common, v - common)), measured, window_ids)
                               for v in (a, b)],
    }


def ordered_scores(observed, modeled, precursor, energies):
    """Average independent scan evidence, in query-side × structure-side order."""
    if (len(observed) != 2 or len(modeled) != 2 or not energies or precursor <= 0
            or any(len(side) != len(energies) for side in (*observed, *modeled))):
        raise ValueError('Expected two candidates and two parallel series of energy-matched scans')
    result = {arm: np.zeros((2, 2)) for arm in ARMS}
    for scan, energy in enumerate(energies):
        pair = [candidate[scan] for candidate in modeled]
        for row, query_side in enumerate(observed):
            query = query_side[scan]
            result['cosine'][row] += [evidence(query, candidate, precursor, energy)[0] for candidate in pair]
            for arm, values in reconstruction(query, pair, precursor).items():
                result[arm][row] += values
    return {arm: values / len(energies) for arm, values in result.items()}
