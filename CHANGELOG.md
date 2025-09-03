## [2.0.0] - 2025-09-03

### Added

- New scikit-learn compatible API
- New user-frendly `fame3r` command
- Support for integration into the NERDD webserver
- Full API and CLI documentation

### Changed

- Full rework of the internals
- Old scripts are no longer supported, use the `fame3r` command instead

### Fixed

- Fixed fingerprint descriptor generation
- Improved FAME score calculation performance

## [1.0.3] - 2025-05-19

### Added

- Added important usage documentation

### Changed

### Fixed

- Improved formatting/linting

## [1.0.2] - 2025-05-19

### Added

- Optional computation of FAME scores via `--compute-fame-scores` flag in test and infer scripts

### Changed

- Optimal default hyperparameters according to results observed on latest MetaQSAR data
- harmonized output formatting (predictions -> y_prob, predictions_binary -> y_pred)

### Fixed
