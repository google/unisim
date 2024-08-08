## [1.0.1] - 2024-08-07

### Added

- Domain phishing detection colab

### Changed

- Reset index after `match()` as it supposed to be stateless
- Added option to return a raw ResultCollection out of `match()`

### Fixed

- Fixed GPU support for onnx which was disabled early on due to
bug in the runtime that appears to be fixed
- Debug messages now uses logging to make UniSim easier to use cli

## [1.0.0] - 2024-05-21

Initial release