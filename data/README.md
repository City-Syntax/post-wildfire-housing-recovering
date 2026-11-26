# Numerical inputs

The analysis entry points read the files under `data/processed/` and the source workbooks under `data/raw/`.

| Input | Coverage and use |
|---|---|
| `frozen_sample.parquet` | 4,637 buildings: 4,614 in California and 23 in Colorado. Main recovery outcomes and model covariates. |
| `building_inventory.parquet` | 56,178 inventory records; the matched California 2014–2025 destroyed-building candidate subset contains 47,653 records. |
| `legacy_analysis_frame.parquet` | 4,639-row descriptive/transaction source frame, including two New Mexico records. Used by the continuous summaries in Supplementary Table 2, income-group summaries in Supplementary Table 4 and transaction preparation. |
| `label_records.csv` | Classification records and model identifiers used to recover the established label priority. |
| Vegetation measurements | Per-image and per-building SegFormer/HSV fractions. |
| Policy and tract inputs | Building-linked CDI policy data, HMDA measures, ACS insurance proxy and forward fire-risk measures. |
| `geography/` | Census 2022 county, tract and place boundaries used by the maps. |

Figure 2 uses 4,310 classifiable observations from the specified fire-age cohort; its adjusted curve uses 4,121 complete cases. Table 1 uses 4,186 rebuilding observations and 1,614 rebuilt observations for upgrading. The joint policy model uses 2,475 complete cases. Supplementary structure, observation-time and fixed-effect analyses retain their stated model-specific samples.

Recovery summaries use state/incident-name groups. Inferential event identifiers and actual observation times are retained in the numerical sample. The vegetation figure uses the source inventory-income tertiles; the rebuilding and policy group summaries use USD 60,000 and USD 100,000 boundaries. FAIR Plan share is stored in percentage units; aggregate non-renewal and HMDA denial rates are fractions.

Owner-name, postal-address and assessor parcel-number columns are excluded from these prepared inputs. Field descriptions are in `DATA_DICTIONARY.csv`. The input manifest records file hashes; normal runs validate them before calculating results. Source URLs and hashes for geographical boundary products are recorded in `geography/source_manifest.json`.
