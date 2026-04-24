# Contributing to Bedrock Smart Router

Thank you for your interest in contributing! This document provides guidelines for contributing to the project.

## Getting Started

1. Fork the repository
2. Clone your fork:
   ```bash
   git clone https://github.com/YOUR_USERNAME/bedrock-smart-router.git
   cd bedrock-smart-router
   ```
3. Set up the development environment:
   ```bash
   python -m venv .venv
   source .venv/bin/activate
   pip install -e ".[dev,redis,otel,faiss]"
   ```
4. Create a branch for your change:
   ```bash
   git checkout -b feature/my-change
   ```

## Running Tests

```bash
# Unit tests (no AWS credentials needed)
pytest tests/ -v

# Integration tests (requires AWS credentials)
INTEGRATION_TEST=1 pytest tests/ -v -s

# Valkey tests (requires VPC access + endpoint)
INTEGRATION_TEST=1 VALKEY_URL=rediss://your-endpoint:6379 pytest tests/test_valkey_cache_integration.py -v -s

# Guardrail tests (requires a guardrail ID)
INTEGRATION_TEST=1 GUARDRAIL_ID=your-id pytest tests/test_guardrails_real_integration.py -v -s
```

## Code Style

- Python 3.11+ with type hints
- Use `from __future__ import annotations` in every module
- Docstrings for all public classes and methods
- No external dependencies in the core SDK (only `boto3`)
- Optional dependencies go in extras (`[redis]`, `[otel]`, `[faiss]`)

## Pull Request Process

1. Ensure all unit tests pass (`pytest tests/ -v`)
2. Add tests for new functionality
3. Update documentation (README, GUIDE.md, iam-permissions.md) if applicable
4. Update `models.json` if adding new models — validate with `python scripts/refresh_pricing.py`
5. Keep commits focused — one logical change per commit
6. Write clear commit messages describing what and why

## What to Contribute

**Welcome contributions:**
- Bug fixes with test cases
- New model entries in `data/models.json` (with pricing validation)
- New routing strategy plugins
- Documentation improvements
- Performance optimizations with benchmarks
- New vector store backends for semantic cache

**Please discuss first (open an issue):**
- New core features that change the routing pipeline
- Changes to the `RoutingDecision` dataclass (breaking change)
- New required dependencies
- Changes to the config schema

## Reporting Issues

- Use GitHub Issues
- Include: Python version, boto3 version, AWS region, error traceback
- For routing issues: include the `routing_decision` output
- For pricing issues: include output from `scripts/refresh_pricing.py`

## License

By contributing, you agree that your contributions will be licensed under the MIT License.
