# Contributing to YOLOv5 Node

## Setup local development environment

For some IDEs it is beneficial to have a .venv environment in the root of the repository.
It is recommended to use the env coniguration of the detector_cpu sub-project because it can be installed on macOS.

```bash
UV_PROJECT_ENVIRONMENT=$(pwd)/.venv uv sync --project ./detector_cpu
```

Then set it as the project interpreter in your IDE (e.g. Python: Select Interpreter -> Workspace -> Enter path to .venv).

## Publish a Release

To publish a new release of the software (trainer and detectors), create a new release via github and use the [version number](https://semver.org/) as name and tag name with the pattern `v<MAJOR>.<MINOR>.<PATCH>`.
This will automatically trigger a pipeline action that builds and publishes docker containers as defined in the [workflow file](.github/workflows/docker-deploy.yml).

## Vendored Upstream Code

[detector/tensorrtx](detector/tensorrtx) is a copy of [wang-xinyu/tensorrtx](https://github.com/wang-xinyu/tensorrtx) (trt10 branch).
Avoid changing it, so that upstream patches can still be merged.

If a change is unavoidable, mark it with a `PATCH (yolov5-node)` comment at the place it applies, so that `grep -rn "PATCH (yolov5-node)"` lists every deviation.

## Data Team Standards

## Linting

We use [ruff](https://docs.astral.sh/ruff/) to enforce the coding style.
Its rules live in the `pyproject.toml` of the respective (sub-)project.

Repositories that ship a `.pre-commit-config.yaml` run ruff through
[pre-commit](https://github.com/pre-commit/pre-commit).
Install the git hooks once:

```bash
python3 -m pip install pre-commit
pre-commit install
```

After that you can check the whole repository with:

```bash
pre-commit run --all-files
```

Repositories without a `.pre-commit-config.yaml` run ruff directly:

```bash
uvx ruff check .
```

The repository-specific part of this file names the exact command wherever it differs.

## Style Guide

### 1. Single or double quotes

We use single quotes for strings in the code and double quotes for docstrings.
Double quotes may also be used for strings if the string contains a single quote.

### 2. String formatting

We use f-strings if possible (due to their readability).
Logs shall use the lazy formatting provided by logging (performance optimization).
When diverging from above rules, please provide a short comment with `# NOTE: ...` to explain the reason.

### 3. Line continuation

To break a long statement across multiple lines, we use a trailing backslash (`\`) to break between chained calls, rather than breaking inside a call's parentheses. This reads better for
fluent/builder chains such as NiceGUI element construction, where the leading `.` on each line clearly marks the continuation and each call keeps its arguments together.

```python
# preferred
ui.button('Speichern', on_click=self.save)\
    .props('no-caps')\
    .classes('button-primary')

# avoid (using the open parenthesis of a call to wrap)
ui.button('Speichern', on_click=self.save).props(
    'no-caps').classes(
    'button-primary')
```

### 4. Comments

We keep comments as few and as short as possible.
A comment should only state what is needed to understand the code — not the motivation behind an implementation and not a restatement of what the code already says.
If the code is clear on its own, it needs no comment.

### 5. Error handling

We do not use return values to signal success or failure.
A function that returns `True`/`False` or `None` for this purpose forces every caller to remember the convention, and a forgotten check fails silently.
Instead we raise exceptions — custom exception classes where appropriate — and handle them where the caller can react.

```python
# preferred
def parse_config(path: Path) -> Config:
    if not path.exists():
        raise ConfigNotFoundError(path)
    ...

# avoid (None as failure signal)
def parse_config(path: Path) -> Config | None:
    if not path.exists():
        return None
    ...
```

### 6. Ordering: Important things first

We want to have main classes and functions at the top of the file, while helper functions and classes should be placed below. This allows to quickly understand the main purpose of the file without having to scroll through a lot of code.

The same rule applies within a class:

- The public interface comes first: the constructor, then the public methods and properties.
- Every private helper is placed below the methods that call it, so a reader always meets the caller before the callee.
- If a helper is called by several methods, it is placed below all of them.

```python
class ReportGenerator:

    def __init__(self, source: Path) -> None:
        self.source = source

    def generate_summary(self) -> str:
        rows = self._load_rows()
        ...

    def generate_details(self) -> str:
        rows = self._load_rows()
        ...

    def _load_rows(self) -> list[Row]:
        """Called by generate_summary and generate_details, so it comes after both."""
        ...
```

### 7. Docstrings and type hints

We use the reStructuredText (reST) or Sphinx-style docstring format.
We don't declare types in the docstring but use type hints.
While type hints are mandatory, parameter descriptions and docstrings should only
be used if the function and parameter names are not self-explanatory.
We write them in the modern annotation style — `list[x]`, not `List[x]` — and import from
`collections.abc` rather than `typing` wherever both offer the name (e.g. `Sequence`).
Examples are listed below:

#### One-line Docstring without parameters

```python
def greet() -> None:
    """Print a friendly greeting."""
    print('Hello!')
```

#### One-line Docstring with parameters

```python
def square(x: int | float) -> int | float:
    """Return the square of x.

    :param x: Number to square.
    :return: The squared value.
    """
    return x * x
```

#### Multi-line Docstring without parameters

```python
def get_timestamp() -> str:
    """
    Return the current timestamp as a formatted string.

    This function retrieves the current local time and formats it
    as "YYYY-MM-DD HH:MM:SS". It can be useful for logging or
    displaying time information in user interfaces.
    """
    return datetime.now().strftime('%Y-%m-%d %H:%M:%S')
```

#### Multi-line Docstring with parameters

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
    :raises FileExistsError: If the file exists and overwrite is False.
    """
    if os.path.exists(filename) and not overwrite:
        raise FileExistsError(f'{filename} already exists.')
    mode = 'wb' if isinstance(data, bytes) else 'w'
    with open(filename, mode) as f:
        f.write(data)
```

### 8. Relative imports within a package

Inside a package we import other modules of the same package with relative imports, not with the package name.
This way the package can be renamed or moved without touching its imports.
We use the shortest path: siblings with a single dot, no going up and back down (`from ..ui.numpad import ...` in a module that lives in `ui/` itself).
Code outside the package — scripts, tests — has no choice and uses absolute imports.

```python
# preferred (in app/ui/scan_view.py)
from .. import config
from ..system import System
from .numpad import Numpad

# avoid
from app import config
from app.system import System
from app.ui.numpad import Numpad
```

### 9. Dataclasses and constructor defaults

We declare dataclasses with `kw_only=True` and `slots=True`, so every field is named at the call site
and typos become errors instead of new attributes.

```python
# preferred
@dataclass(kw_only=True, slots=True)
class Detection:
    category: str
    confidence: float

# avoid
@dataclass
class Detection:
    category: str
    confidence: float
```

Be conservative with default `__init__` parameters.
A parameter that no caller ever varies is a constant, not a parameter, and hiding it in the signature
suggests a flexibility that nobody uses.

### 10. Logging and debug output

We log through the module-level logger, never through the root logger, and with lazy formatting, so
the arguments are only rendered when the record is actually emitted.
For temporary debug output we use `print` with `flush=True`, so the line shows up in the log right
away instead of sitting in the buffer.

```python
# preferred
logger = logging.getLogger('mypackage.mymodule')
logger.info('loaded %s images', count)
print(f'reached the empty-category branch with {count} images', flush=True)

# avoid
logging.info(f'loaded {count} images')
print(f'reached the empty-category branch with {count} images')
```
