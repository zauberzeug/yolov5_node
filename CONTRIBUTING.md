# Contributing to YOLOv5 Node

## Setup local development environment

For some IDEs it is beneficial to have a .venv environment in the root of the repository.
It is recommended to use the env coniguration of the detector_cpu sub-project because it can be installed on macOS.

```bash
UV_PROJECT_ENVIRONMENT=$(pwd)/.venv uv sync --project ./detector_cpu
```

Then set it as the project interpreter in your IDE (e.g. Python: Select Interpreter -> Workspace -> Enter path to .venv).

## Publish a Release

To publish a new release of the software (trainer and detectors), create a new release via github and use the [version number](https://semver.org/) as a name and tag name with the pattern `v<MAJOR>.<MINOR>.<PATCH>`.
This will automatically trigger a pipeline action that builds and publishes docker containers as defined in the [workflow file](.github/workflows/docker-deploy.yml).

## Style Guide

### Single or double quotes

We use single quotes for strings in the code and double quotes for docstrings.
Double quotes may also be used for strings if the string contains a single quote.

### String formatting

We use f-strings if possible (due to their readability).
Logs shell use the lazy formatting provided by the logging (performance optimization).
When diverging from above rules, please provide a short comment with `# NOTE: ...` to explaining the reason.

## Docstrings and type hints

We use the reStructuredText (reST) or Sphinx-style docstring format.
We don't declare types in the docstring but use type hints.
While type hints are mandatory, parameter descriptions and docstrings should only
be used if the function and parameter names are not self-explanatory.
Examples are listetd below:

### One-line Docstring without parameters

```python
def greet() -> None:
    """Print a friendly greeting."""
    print("Hello!")
```

### One-line Docstring with parameters

```python
def square(x: int | float) -> int | float:
    """Return the square of x.

    :param x: Number to square.
    :return: The squared value.
    """
    return x * x
```

### Multi-line Docstring without parameters

```python
def get_timestamp() -> str:
    """
    Return the current timestamp as a formatted string.

    This function retrieves the current local time and formats it
    as "YYYY-MM-DD HH:MM:SS". It can be useful for logging or
    displaying time information in user interfaces.
    """
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")
```

### Multi-line Docstring with parameters

```python
def save_data(data: str | bytes, filename: str, overwrite: bool = False) -> None:
    """
    Save data to a file.

    This function writes the provided data to a file on disk.
    If the file already exists and `overwrite` is False, an
    exception is raised to prevent accidental data loss.

    :param data: The content to be saved. Can be text or bytes.
    :param filename: Path to the destination file.
    :param overwrite: Whether to overwrite an existing file.
    :return: True if the data was saved successfully, False otherwise.
    :raises FileExistsError: If the file exists and overwrite is False.
    """
    if os.path.exists(filename) and not overwrite:
        raise FileExistsError(f"{filename} already exists.")
    mode = "wb" if isinstance(data, bytes) else "w"
    with open(filename, mode) as f:
        f.write(data)
```
