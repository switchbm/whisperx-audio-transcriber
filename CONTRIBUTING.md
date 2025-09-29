# Contributing to WhisperX Audio Transcriber

We love your input! We want to make contributing to this project as easy and transparent as possible, whether it's:

- Reporting a bug
- Discussing the current state of the code
- Submitting a fix
- Proposing new features
- Becoming a maintainer

## Development Process

We use GitHub to host code, track issues and feature requests, and accept pull requests.

## Pull Requests

Pull requests are the best way to propose changes to the codebase. We actively welcome your pull requests:

1. Fork the repo and create your branch from `main`.
2. If you've added code that should be tested, add tests.
3. If you've changed APIs, update the documentation.
4. Ensure the test suite passes.
5. Make sure your code lints.
6. Issue that pull request!

## Development Setup

```bash
# Clone your fork
git clone https://github.com/yourusername/whisperx-audio-transcriber.git
cd whisperx-audio-transcriber

# Create virtual environment
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install in development mode
pip install -e ".[dev]"

# Set up pre-commit hooks (optional but recommended)
pre-commit install
```

## Testing

We use pytest for testing:

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=. tests/

# Run specific test file
pytest tests/test_transcriber.py

# Run with verbose output
pytest -v
```

## Code Style

We use several tools to maintain code quality:

- **Black** for code formatting
- **isort** for import sorting
- **flake8** for linting

Run these before submitting:

```bash
# Format code
black .
isort .

# Check linting
flake8 .

# Or run all at once
black . && isort . && flake8 .
```

## Reporting Bugs

We use GitHub issues to track public bugs. Report a bug by [opening a new issue](https://github.com/yourusername/whisperx-audio-transcriber/issues).

**Great Bug Reports** tend to have:

- A quick summary and/or background
- Steps to reproduce
  - Be specific!
  - Give sample code if you can
- What you expected would happen
- What actually happens
- Notes (possibly including why you think this might be happening, or stuff you tried that didn't work)

## Feature Requests

Feature requests are welcome! Please provide:

- **Clear description** of the feature
- **Use case** - why would this be useful?
- **Proposed implementation** (if you have ideas)
- **Alternatives considered**

## Priority Areas for Contributions

### 🏆 High Impact
- **Performance optimizations** for large files
- **Memory usage improvements**
- **Better error handling** for edge cases
- **Multi-language testing** and improvements

### 🎯 Medium Impact
- **Web interface** (Flask/FastAPI/Gradio)
- **GUI application** (tkinter/PyQt)
- **Docker containerization**
- **Additional output formats**

### 💡 Great for Beginners
- **Documentation improvements**
- **Example scripts** for common use cases
- **Test coverage** improvements
- **Code comments** and docstrings

## Development Guidelines

### Code Organization
- Keep functions focused and single-purpose
- Use meaningful variable and function names
- Add docstrings to public functions
- Handle errors gracefully with informative messages

### Performance
- Profile code for performance bottlenecks
- Consider memory usage for large files
- Cache expensive operations when possible
- Use appropriate data structures

### Compatibility
- Support Python 3.8+
- Test on multiple operating systems
- Consider both CPU and GPU environments
- Handle missing dependencies gracefully

## Testing Guidelines

### Test Types
- **Unit tests** for individual functions
- **Integration tests** for complete workflows
- **Performance tests** for large files
- **Error handling tests** for edge cases

### Test Data
- Use small test audio files (<1MB)
- Create reproducible test scenarios
- Test with different audio formats
- Include multi-speaker test cases

## Documentation

### Code Documentation
- Use clear docstrings with parameter types
- Include usage examples in docstrings
- Comment complex algorithms
- Keep README.md updated with new features

### User Documentation
- Update CLI help text for new options
- Add new features to README examples
- Create guides for common use cases
- Include troubleshooting tips

## Release Process

1. Update version in `pyproject.toml`
2. Update CHANGELOG.md with new features
3. Ensure all tests pass
4. Create release PR
5. Tag release after merge
6. Update documentation

## Code of Conduct

### Our Pledge
We are committed to making participation in this project a harassment-free experience for everyone.

### Our Standards
- Using welcoming and inclusive language
- Being respectful of differing viewpoints
- Gracefully accepting constructive criticism
- Focusing on what is best for the community
- Showing empathy towards other community members

## Questions?

Don't hesitate to ask questions! You can:

- Open a [GitHub issue](https://github.com/yourusername/whisperx-audio-transcriber/issues)
- Start a [discussion](https://github.com/yourusername/whisperx-audio-transcriber/discussions)
- Reach out to maintainers

## License

By contributing, you agree that your contributions will be licensed under the MIT License.