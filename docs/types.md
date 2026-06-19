# Types

## Inputs

Python accepts:

- Lists: `[[1, 2], [3, 4]]`
- NumPy arrays when NumPy is installed
- pandas DataFrames when pandas is installed
- Row dictionaries: `{"rooms": 2, "sqft": 950}`
- Lists of row dictionaries

C++ accepts `simplennet::Matrix`, which is `std::vector<std::vector<double>>`.

The C ABI accepts row-major `double*` input arrays.

## Outputs

Output types:

- `int`: regression output rounded to native integers
- `float`: regression output returned as floating point
- `category`: softmax classification returned as the original string label

Python can infer the output type from targets, or callers can pass `int`, `float`, or `"category"` explicitly.
