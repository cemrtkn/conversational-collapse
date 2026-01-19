# babel-ai

A research framework for analyzing long-term behavior and drift patterns in Large Language Models (LLMs) during self-loop conversations without external input.

## Overview

This project investigates whether AIs need external input to create sensible output and explores the importance of human interaction for AI performance. It analyzes how AI interaction can be made more diverse without external input and what productive input patterns look like.

## Quick Start

### Installation

1. **Prerequisites**: Python 3.13+ and uv

   **Install uv if needed**:
   ```bash
   # Official installer (recommended)
   curl -LsSf https://astral.sh/uv/install.sh | sh
   ```

2. **Clone and install**:
   ```bash
   # Clone the repository
   git clone <repository-url>
   cd conversational-collapse

   # Install dependencies
   uv sync

   # Activate the virtual environment
   source .venv/bin/activate
   ```

3. **Set up environment** (optional):
   ```bash
   # Create .env file
   cp .env.example .env  # Edit huggingface-related info
   ```

### Running Tests

```bash
# Run all tests
uv run pytest

# Run with coverage
uv run coverage run --source=src -m pytest
uv run coverage report

# Run specific test categories
uv run pytest tests/unit_tests/     # Unit tests only
uv run pytest tests/integration/   # Integration tests only
```

### Quick Experiment

1. **Basic experiment**:
   ```bash
   # Run with the provided test configuration
   sbatch scripts/run_experiment.sh
   ```

2. **Custom experiment**:
   ```bash
   # Create your own config file (see configs/test_config.yaml as template)
   sbatch scripts/run_experiment.sh debug <path_to_new_config>
   ```

3. **View results**:
   - Results saved to `results/` directory as CSV and JSON files
   - Use notebooks in `notebooks/` for analysis and visualization

**For detailed experiment configuration, multi-agent setups, and analysis workflows, see [doc/experiments.md](doc/experiments.md).**

## Key Features

- **Inference Allowing Interpretability Experiments**: Built on top of NNSight
- **Drift Analysis**: Semantic similarity, lexical analysis, perplexity metrics
- **Data Sources**: ShareGPT, Topical Chat, Infinite Conversation datasets
- **Configurable Experiments**: YAML-based configuration system
- **Analysis Notebooks**: Jupyter notebooks for result visualization
- **Comprehensive Testing**: Unit and integration test suites

## Project Structure

```
babel_ai/
├── src/
│   ├── babel_ai/          # Core experiment framework
│   ├── api/               # LLM interface
│   ├── models/            # Data models and configurations
│   └── main.py            # Main experiment runner
├── tests/                 # Test suites
├── notebooks/             # Analysis and visualization
├── configs/               # Experiment configurations
├── doc/                   # Detailed documentation
└── data/                  # Datasets (not included)
```

## Contributing

We welcome contributions! Please see [doc/contributing.md](doc/contributing.md) for detailed guidelines including:
- Code style and formatting (Black, 79-char limit)
- Testing requirements (pytest, coverage)
- Development workflow
- Adding new analyzers, fetchers, or providers

## Documentation

- **[Experiments Guide](doc/how_to_run_experiments.md)**: How to run experiments and use notebooks
- **[Contributing Guide](doc/contributing.md)**: Development guidelines

## License

[LICENSE](LICENSE)

## Research Context

This project explores fundamental questions about AI behavior:
- Do AIs require external input for coherent output?
- How important is human interaction for AI performance?
- Can AI interaction diversity be improved without external input?
- What input patterns are most productive for AI systems?
