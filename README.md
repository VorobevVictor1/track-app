# tracks-app

[![Python Version](https://shields.io)](https://python.org)
[![License: MIT](https://shields.io)](https://opensource.org)

Application for analysis of fission tracks in mineral grains.

## Installation

This project uses **uv** as its package and project manager.

### 1. Prerequisites

Make sure you have `uv` installed. If not, install it using one of the following commands:

*   **macOS / Linux:**
    ```bash
    curl -LsSf https://astral.sh/uv/install.sh | sh
    ```
*   **Windows (PowerShell):**
    ```powershell
    powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
    ```

### 2. Clone the Repository

```bash
git clone github.com/VorobevVictor1/track-app
cd track-app
```

### 3. Setup Environment & Dependencies

Install the project, setup the virtual environment, and install production dependencies:

```bash
uv sync
```

#### For Developers

If you want to run tests or code linters, install the development dependencies as well:

```bash
uv sync --group dev
```

### 4. Enable Pre-commit Hooks

To maintain code standards before committing changes, activate `pre-commit`:

```bash
uv run pre-commit install
```

## Usage

To run the application or scripts securely within the managed environment, prefix your commands with `uv run`:

```bash
# Example: run the camera check utility
uv run python camera_check.py
```

## Development & Quality Tools

- **Lint & Format Code:** `uv run ruff check` / `uv run ruff format`
- **Type Checking:** `uv run mypy src`
- **Run Tests:** `uv run pytest`

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
