use apbase::{Map, MapOptions};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let text = std::fs::read_to_string(concat!(env!("CARGO_MANIFEST_DIR"), "/tests/data/meuse.csv"))?;
    let rows: Vec<Vec<&str>> = text
        .lines()
        .skip(1)
        .map(|line| line.split(',').collect())
        .collect();
    let column = |index: usize| -> Vec<f64> { rows.iter().map(|row| row[index].parse().unwrap()).collect() };
    let (x, y) = (column(0), column(1));
    let z: Vec<f64> = column(5).iter().map(|zinc| zinc.ln()).collect();

    apbase::config.set("n_threads", 4)?;

    let result = Map::new(MapOptions::new(10.0)).call(&x, &y, &z)?;
    println!("method: {}", result.method);
    println!("compact points: {}", result.z.len());

    let (raster_x, _raster_y, raster_z) = result.to_raster()?;
    println!(
        "raster: {} x {} (NaN outside the boundary)",
        raster_x.ncols(),
        raster_x.nrows()
    );
    println!(
        "finite cells: {}",
        raster_z.iter().filter(|v| v.is_finite()).count()
    );

    let same = apbase::create_map(&x, &y, &z, MapOptions::new(10.0))?;
    println!(
        "create_map gives the same method: {}",
        same.method == result.method
    );
    Ok(())
}
