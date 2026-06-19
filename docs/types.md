# Types

## Inputs

Python accepts:

- Lists: `[[1, 2], [3, 4]]`
- NumPy arrays when NumPy is installed
- pandas DataFrames when pandas is installed
- Row dictionaries: `{"rooms": 2, "sqft": 950}`
- Lists of row dictionaries
- Mixed Python objects through the convenience universal encoder

C++ accepts `simplennet::Matrix`, which is `std::vector<std::vector<double>>`.

The C ABI accepts row-major `double*` input arrays.

## Universal Text Inputs

Every supported runtime also accepts serialized text inputs. Use this path when your source value is an object, event, payload, message, custom struct, or any other datatype that is easier to represent as JSON than as hand-built numeric features.

```json
{"kind":"cart","items":4,"coupon":true}
```

The native core hash-encodes text into a fixed-width numeric vector. The encoder width is stored in `.snet` metadata, so saved text models can be loaded and predicted from another supported language.

Use stable serialization:

- Keep field names consistent.
- Prefer JSON with deterministic key ordering when possible.
- Keep units and enum values consistent between training and prediction.

## Outputs

Output types:

- `int`: regression output rounded to native integers
- `float`: regression output returned as floating point
- `category`: softmax classification returned as the original string label

Python can infer the output type from targets, or callers can pass `int`, `float`, or `"category"` explicitly.
