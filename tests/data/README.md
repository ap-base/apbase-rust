# Reference datasets for cokriging validation

Real-world fixtures used by `tests/test_cokriging_reference_datasets.rs` to
validate `apbase::cokriging` against published geostatistics benchmarks, in
addition to the synthetic tests in `tests/test_cokriging.rs`.

## jura_prediction.csv / jura_validation.csv

Goovaerts (1997), *Geostatistics for Natural Resources Evaluation*, Oxford
University Press — the canonical cokriging benchmark: 7 heavy metals (Cd, Co,
Cr, Cu, Ni, Pb, Zn) in the topsoil of the Swiss Jura, split into 259 training
points (`prediction.dat`) and 100 held-out points (`validation.dat`). `Xloc`/
`Yloc` are local grid coordinates in km.

- Source: `prediction.dat` / `validation.dat` objects inside
  `https://raw.githubusercontent.com/cran/gstat/master/data/jura.rda`
  (CRAN mirror of the `gstat` R package, GPL (>= 2)).
- Regenerate: convert the `prediction.dat` / `validation.dat` objects to CSV,
  e.g. in R: `load("jura.rda"); write.csv(prediction.dat, "jura_prediction.csv", row.names = FALSE)`.

## meuse.csv

Classic Meuse river dataset (Burrough & McDonnell, via the `sp`/`gstat` R
packages): 155 points, heavy metal concentrations plus continuous covariates
`elev` (elevation) and `dist` (distance to the Meuse river). Coordinates
`x`/`y` are already in the Dutch RD projected grid (meters).

- Source: `meuse` object inside
  `https://raw.githubusercontent.com/cran/sp/master/data/meuse.rda`
  (CRAN mirror of the `sp` R package, GPL (>= 2)).
- Regenerate: same `.rda` to CSV conversion as above.

## walker_sample.csv / walker_exhaustive.csv

Walker Lake dataset from Isaaks & Srivastava (1989), *Applied Geostatistics*,
Oxford University Press — 470 sparse samples of `v`/`u` (`u` missing at 195
locations, encoded as empty) plus a 78,000-point exhaustive grid with the true
`u`/`v` values, used to check cokriging accuracy (predict `v` from `u`)
against known ground truth rather than only a held-out subset.

- Source: `opengeostat/pygslib` (MIT License) —
  `pygslib/data/WalkerLake/walker.dat` (GSLIB text format, header stripped)
  and `pygslib/data/WalkerLake/Exhaustive_set.csv`. That repo's own readme
  traces the data to the ai-geostats/52North `WalkerLake_01.zip` teaching
  dataset.
- Regenerate: `walker.dat` is whitespace-separated after an 8-line GSLIB
  header, with columns `id, x, y, v, u, t`; values `>= 1e30` are the GSLIB
  missing-data sentinel and are written as empty cells. `Exhaustive_set.csv`
  is plain CSV, only lower-cased here.

## Study-area boundaries: meuse_area.wkt / meuse_grid.csv / jura_area.wkt / walker_area.wkt

Real interpolation domains for the datasets above, used as `bounds` by
`tests/test_reference_boundaries.rs` and compared with the concave hull that
`apbase` infers when no boundary is given. Each `.wkt` holds one `POLYGON` in
the same coordinates as the matching CSV.

- `meuse_area.wkt`: the published study-area outline (391 vertices, Dutch RD
  meters), from the `meuse.area` object inside
  `https://raw.githubusercontent.com/cran/sp/master/data/meuse.area.rda`
  (GPL (>= 2)). Covers all 155 points of `meuse.csv`.
- `meuse_grid.csv`: the official 40 m prediction grid (3103 cell centers;
  columns `x, y, dist, ffreq, soil`), from the `meuse.grid` object inside
  `https://raw.githubusercontent.com/cran/sp/master/data/meuse.grid.rda`.
  Every cell center lies inside `meuse_area.wkt`.
- `jura_area.wkt`: **derived**, not published -- `gstat` ships no Jura polygon.
  It is the union of the 0.05 km cells centered on the 5957 points of the
  official prediction grid `juragrid.dat` (inside the same `jura.rda`), in local
  km. Covers all 359 training/validation points.
- `walker_area.wkt`: the rectangle `0.5..260.5 x 0.5..300.5` drawn by the cells
  of `walker_exhaustive.csv` (`x` 1..260, `y` 1..300), written by hand.
- Regenerate: read the `.rda` objects (e.g. in R, `load("meuse.area.rda")`),
  write the `meuse.area` matrix as a closed `POLYGON`, the `meuse.grid` columns
  above as CSV, and the union of `x +/- 0.025`, `y +/- 0.025` squares around
  each `juragrid.dat` point as a `POLYGON`.

## Notes

These CSVs are prepared once, offline, and committed as static fixtures so
the tests run deterministically without a network dependency. The conversion
tools above are one-off helpers, **not** runtime or test dependencies of
`apbase`.
