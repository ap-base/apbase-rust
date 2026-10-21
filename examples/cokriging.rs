//! The README "Cokriging" example on the Meuse dataset: zinc (155 samples)
//! estimated on the 40 m Meuse grid, first with the elevation and the
//! distance to the river measured at the same samples, then with the
//! distance to the river as a raster (the grid itself).
//!
//!     cargo run --release --example cokriging

use apbase::cokriging::{CoKrigingMethod, CoKrigingOptions, SecondaryInput, co_kriging};
use apbase::cross_variogram::{ScreeningOptions, VariableXYZ, screen_secondary_variables};
use indexmap::IndexMap;
use ndarray::Array1;

/// Columns of a CSV file under `tests/data`, by header name.
fn read_columns(name: &str) -> Result<IndexMap<String, Vec<f64>>, Box<dyn std::error::Error>> {
    let path = format!("{}/tests/data/{name}", env!("CARGO_MANIFEST_DIR"));
    let text = std::fs::read_to_string(path)?;
    let mut lines = text.lines();
    let header: Vec<&str> = lines.next().ok_or("empty file")?.split(',').collect();
    let mut columns: IndexMap<String, Vec<f64>> =
        header.iter().map(|h| (h.to_string(), Vec::new())).collect();
    for line in lines {
        for (cell, column) in line.split(',').zip(columns.values_mut()) {
            column.push(cell.parse().unwrap_or(f64::NAN));
        }
    }
    Ok(columns)
}

fn xyz(x: &[f64], y: &[f64], z: &[f64]) -> VariableXYZ {
    (
        Array1::from(x.to_vec()),
        Array1::from(y.to_vec()),
        Array1::from(z.to_vec()),
    )
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let samples = read_columns("meuse.csv")?;
    let grid = read_columns("meuse_grid.csv")?;
    let (x, y) = (&samples["x"], &samples["y"]);
    let zinc: Vec<f64> = samples["zinc"].iter().map(|v| v.ln()).collect();

    let candidates: IndexMap<String, VariableXYZ> = IndexMap::from([
        ("elev".to_string(), xyz(x, y, &samples["elev"])),
        ("dist".to_string(), xyz(x, y, &samples["dist"])),
    ]);
    let screening = screen_secondary_variables(x, y, &zinc, &candidates, ScreeningOptions::default())?;
    for (name, report) in &screening.report {
        println!(
            "{name}: rho = {:.3}, p = {:.2e}, selected = {}",
            report.rho, report.p_value, report.decision
        );
    }

    let targets: Vec<[f64; 2]> = grid["x"].iter().zip(&grid["y"]).map(|(a, b)| [*a, *b]).collect();
    let estimates = co_kriging(
        x,
        y,
        &zinc,
        &SecondaryInput::from(screening.selected),
        &targets,
        CoKrigingOptions {
            method: CoKrigingMethod::Collocated,
            ..Default::default()
        },
    )?;
    summary("collocated cokriging", &estimates);

    // Raster secondary: no grid cell lies on a sample, so each sample is
    // paired with its nearest cell within half the cell diagonal (28.3 m).
    // The square root makes its relation with ln(zinc) close to linear.
    let sqrt_dist: Vec<f64> = grid["dist"].iter().map(|v| v.sqrt()).collect();
    let raster = SecondaryInput::from([("sqrt_dist", xyz(&grid["x"], &grid["y"], &sqrt_dist))]);
    let estimates = co_kriging(
        x,
        y,
        &zinc,
        &raster,
        &targets,
        CoKrigingOptions {
            method: CoKrigingMethod::Icm,
            max_colocation_dist: Some(28.3),
            max_secondary_neighbors: Some(8),
            ..Default::default()
        },
    )?;
    summary("ICM cokriging with a raster", &estimates);
    Ok(())
}

fn summary(label: &str, estimates: &Array1<f64>) {
    let finite: Vec<f64> = estimates.iter().copied().filter(|v| v.is_finite()).collect();
    let (min, max) = finite
        .iter()
        .fold((f64::INFINITY, f64::NEG_INFINITY), |(lo, hi), v| {
            (lo.min(*v), hi.max(*v))
        });
    println!(
        "{label}: {} of {} grid cells, ln(zinc) {min:.2} .. {max:.2}",
        finite.len(),
        estimates.len()
    );
}
