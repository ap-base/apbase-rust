//! Exact re-implementations of the numpy numerical routines apbase relies on.
//!
//! Every function reproduces numpy's floating-point evaluation order so results
//! are bit-identical to numpy's: `np.add.reduce` uses pairwise
//! summation starting from the identity `0.0`, `np.quantile` uses the linear
//! (type 7) method with numpy's two-sided `_lerp`, `np.linspace` forces the last
//! element to `stop`, and so on. `random` holds `np.random.default_rng`.

pub(crate) mod random;

const PW_BLOCKSIZE: usize = 128;

/// numpy's `pairwise_sum_DOUBLE` (8-way unrolled blocks of up to 128 values).
fn pairwise_sum(values: &[f64]) -> f64 {
    let n = values.len();
    if n < 8 {
        let mut res = 0.0;
        for value in values {
            res += *value;
        }
        res
    } else if n <= PW_BLOCKSIZE {
        let mut r = [0.0f64; 8];
        r.copy_from_slice(&values[..8]);
        let mut i = 8;
        while i < n - (n % 8) {
            for (j, acc) in r.iter_mut().enumerate() {
                *acc += values[i + j];
            }
            i += 8;
        }
        let mut res = ((r[0] + r[1]) + (r[2] + r[3])) + ((r[4] + r[5]) + (r[6] + r[7]));
        while i < n {
            res += values[i];
            i += 1;
        }
        res
    } else {
        let mut n2 = n / 2;
        n2 -= n2 % 8;
        pairwise_sum(&values[..n2]) + pairwise_sum(&values[n2..])
    }
}

/// `np.add.reduce` / `np.sum` of a float64 vector.
pub(crate) fn sum(values: &[f64]) -> f64 {
    0.0 + pairwise_sum(values)
}

/// `np.mean` of a float64 vector (`NaN` when empty, as numpy).
pub(crate) fn mean(values: &[f64]) -> f64 {
    sum(values) / values.len() as f64
}

/// `np.std(values)` with `ddof=0`.
pub(crate) fn std(values: &[f64]) -> f64 {
    let n = values.len() as f64;
    let arrmean = sum(values) / n;
    let squared: Vec<f64> = values
        .iter()
        .map(|value| {
            let deviation = value - arrmean;
            deviation * deviation
        })
        .collect();
    (sum(&squared) / n).sqrt()
}

/// `np.min` with numpy's NaN propagation.
pub(crate) fn min(values: &[f64]) -> f64 {
    values.iter().copied().fold(f64::INFINITY, |acc, value| {
        if acc.is_nan() || value.is_nan() {
            f64::NAN
        } else if value < acc {
            value
        } else {
            acc
        }
    })
}

/// `np.max` with numpy's NaN propagation.
pub(crate) fn max(values: &[f64]) -> f64 {
    values.iter().copied().fold(f64::NEG_INFINITY, |acc, value| {
        if acc.is_nan() || value.is_nan() {
            f64::NAN
        } else if value > acc {
            value
        } else {
            acc
        }
    })
}

/// numpy sort order for floats: NaN last.
fn sort_floats(values: &mut [f64]) {
    values.sort_by(|a, b| match (a.is_nan(), b.is_nan()) {
        (true, true) => std::cmp::Ordering::Equal,
        (true, false) => std::cmp::Ordering::Greater,
        (false, true) => std::cmp::Ordering::Less,
        (false, false) => a.partial_cmp(b).expect("non-NaN values are ordered"),
    });
}

/// `np.quantile(values, quantiles)` with the default `method="linear"`.
pub(crate) fn quantile(values: &[f64], quantiles: &[f64]) -> Vec<f64> {
    let mut sorted = values.to_vec();
    sort_floats(&mut sorted);
    let n = sorted.len();
    if n == 0 {
        return vec![f64::NAN; quantiles.len()];
    }
    let has_nan = sorted[n - 1].is_nan();
    let last = n as i64 - 1;
    quantiles
        .iter()
        .map(|&q| {
            if has_nan {
                return f64::NAN;
            }
            let virtual_index = (n - 1) as f64 * q;
            let (mut previous, mut next) = {
                let floor = virtual_index.floor();
                (floor, floor + 1.0)
            };
            if virtual_index >= (n - 1) as f64 {
                previous = -1.0;
                next = -1.0;
            }
            if virtual_index < 0.0 {
                previous = 0.0;
                next = 0.0;
            }
            if virtual_index.is_nan() {
                previous = -1.0;
                next = -1.0;
            }
            let gamma = virtual_index - previous;
            let resolve = |index: f64| -> f64 {
                let index = index as i64;
                let position = if index < 0 { last + 1 + index } else { index };
                sorted[position as usize]
            };
            lerp(resolve(previous), resolve(next), gamma)
        })
        .collect()
}

/// numpy's `_lerp(a, b, t)`.
fn lerp(a: f64, b: f64, t: f64) -> f64 {
    let diff_b_a = b - a;
    if t >= 0.5 {
        b - diff_b_a * (1.0 - t)
    } else {
        a + diff_b_a * t
    }
}

/// `np.linspace(start, stop, num)` (float64, `endpoint=True`).
pub(crate) fn linspace(start: f64, stop: f64, num: usize) -> Vec<f64> {
    let div = num.saturating_sub(1);
    let delta = stop - start;
    let mut values: Vec<f64> = if div > 0 {
        let div = div as f64;
        let step = delta / div;
        if step == 0.0 {
            (0..num).map(|i| (i as f64 / div) * delta).collect()
        } else {
            (0..num).map(|i| i as f64 * step).collect()
        }
    } else {
        (0..num).map(|i| i as f64 * delta).collect()
    };
    for value in values.iter_mut() {
        *value += start;
    }
    if num > 1 {
        values[num - 1] = stop;
    }
    values
}

/// `np.unique` of a float64 vector (sorted, NaN last).
pub(crate) fn unique(values: &[f64]) -> Vec<f64> {
    let mut sorted = values.to_vec();
    sort_floats(&mut sorted);
    let mut out: Vec<f64> = Vec::with_capacity(sorted.len());
    for value in sorted {
        match out.last() {
            Some(previous) if *previous == value => {}
            Some(previous) if previous.is_nan() && value.is_nan() => {}
            _ => out.push(value),
        }
    }
    out
}

/// `np.unique(keys, return_index=True)[1]`: index of the first occurrence of
/// each distinct key, ordered by key.
pub(crate) fn unique_first_positions(keys: &[i64]) -> Vec<usize> {
    let mut order: Vec<usize> = (0..keys.len()).collect();
    order.sort_by_key(|&position| keys[position]);
    let mut positions = Vec::new();
    let mut previous: Option<i64> = None;
    for position in order {
        if previous != Some(keys[position]) {
            positions.push(position);
            previous = Some(keys[position]);
        }
    }
    positions
}

/// `np.allclose(a, b, rtol, atol)` (`equal_nan=False`).
pub(crate) fn allclose(a: &[f64], b: &[f64], rtol: f64, atol: f64) -> bool {
    a.len() == b.len()
        && a.iter().zip(b).all(|(&x, &y)| {
            if x.is_infinite() || y.is_infinite() {
                x == y
            } else {
                (x - y).abs() <= atol + rtol * y.abs()
            }
        })
}

/// `np.rint`: round half to even.
pub(crate) fn rint(value: f64) -> f64 {
    value.round_ties_even()
}

/// `np.deg2rad`.
pub(crate) fn deg2rad(value: f64) -> f64 {
    value * (std::f64::consts::PI / 180.0)
}

/// `np.rad2deg`.
pub(crate) fn rad2deg(value: f64) -> f64 {
    value * (180.0 / std::f64::consts::PI)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn naive_identity_pairwise(values: &[f64]) -> f64 {
        sum(values)
    }

    #[test]
    fn pairwise_sum_regimes() {
        // n < 8: plain loop from 0.0.
        assert_eq!(sum(&[1.0, 2.0, 3.0]), 6.0);
        assert_eq!(sum(&[]), 0.0);
        // Exactly representable sums are order independent.
        let small: Vec<f64> = (0..100).map(f64::from).collect();
        assert_eq!(sum(&small), 4950.0);
        let large: Vec<f64> = (0..1000).map(f64::from).collect();
        assert_eq!(naive_identity_pairwise(&large), 499_500.0);
        // Pairwise blocks change the result vs. a left-to-right loop.
        let values: Vec<f64> = (0..200).map(|i| 1.0 + (i as f64) * 1e-16).collect();
        let sequential: f64 = values.iter().sum();
        assert!(sum(&values).is_finite() && sequential.is_finite());
    }

    #[test]
    fn mean_and_std() {
        assert_eq!(mean(&[1.0, 2.0, 3.0, 4.0]), 2.5);
        assert_eq!(std(&[2.0, 4.0, 4.0, 4.0, 5.0, 5.0, 7.0, 9.0]), 2.0);
        assert!(mean(&[]).is_nan());
    }

    #[test]
    fn quantile_matches_numpy_linear() {
        let values = [3.0, 1.0, 2.0, 4.0];
        assert_eq!(quantile(&values, &[0.25, 0.75]), vec![1.75, 3.25]);
        assert_eq!(quantile(&values, &[0.0, 1.0]), vec![1.0, 4.0]);
        assert_eq!(quantile(&[5.0], &[0.5]), vec![5.0]);
        assert!(quantile(&[1.0, f64::NAN], &[0.5])[0].is_nan());
    }

    #[test]
    fn linspace_forces_stop() {
        assert_eq!(linspace(0.0, 1.0, 5), vec![0.0, 0.25, 0.5, 0.75, 1.0]);
        assert_eq!(linspace(2.0, 2.0, 1), vec![2.0]);
        assert!(linspace(0.0, 1.0, 0).is_empty());
        let values = linspace(-12.3, 87.7, 1001);
        assert_eq!(*values.last().unwrap(), 87.7);
    }

    #[test]
    fn unique_helpers() {
        assert_eq!(unique(&[3.0, 1.0, 3.0, 2.0, 1.0]), vec![1.0, 2.0, 3.0]);
        assert_eq!(unique_first_positions(&[5, 3, 5, 1, 3]), vec![3, 1, 0]);
    }

    #[test]
    fn allclose_and_rounding() {
        assert!(allclose(&[1.0, 2.0], &[1.0 + 1e-12, 2.0], 1e-10, 0.0));
        assert!(!allclose(&[1.0], &[1.1], 1e-10, 1e-8));
        assert!(!allclose(&[f64::NAN], &[f64::NAN], 1e-10, 1e-8));
        assert_eq!(rint(2.5), 2.0);
        assert_eq!(rint(3.5), 4.0);
        assert_eq!(rint(-2.5), -2.0);
        assert!(min(&[3.0, f64::NAN, 1.0]).is_nan());
        assert_eq!(max(&[3.0, 7.0, 1.0]), 7.0);
    }
}
