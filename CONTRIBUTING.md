# Contributing to SmartAlert

Thank you for your interest in contributing to SmartAlert! This document provides guidelines and information for contributors.

## Table of Contents

- [Getting Started](#getting-started)
- [Development Setup](#development-setup)
- [Code Style](#code-style)
- [Testing](#testing)
- [Submitting Changes](#submitting-changes)
- [Issue Guidelines](#issue-guidelines)
- [Pull Request Guidelines](#pull-request-guidelines)

## Getting Started

Before contributing, please:

1. Read this contributing guide
2. Check existing issues and pull requests to avoid duplicates
3. Join our discussions in the issues section

## Development Setup

### Prerequisites

- Python 3.8 or higher
- Git
- pip or conda for package management

### Installation

1. Fork the repository
2. Clone your fork:
   ```bash
   git clone https://github.com/YOUR_USERNAME/SmartAlert.git
   cd SmartAlert
   ```

3. Create a virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

4. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

5. Set up pre-commit hooks (optional but recommended):
   ```bash
   pip install pre-commit
   pre-commit install
   ```

## Code Style

### Python

- Follow PEP 8 style guidelines
- Use meaningful variable and function names
- Add docstrings to functions and classes
- Keep functions focused and concise
- Use type hints where appropriate

### Jupyter Notebooks

- Clear cell outputs before committing
- Add descriptive markdown cells
- Keep notebooks focused on specific tasks
- Document assumptions and data sources

### File Organization

- Keep related files together
- Use descriptive file names
- Organize notebooks in the `notebooks/` directory
- Store utilities in the `utils/` directory

## Testing

### Running Tests

```bash
# Run all tests
python -m pytest

# Run tests with coverage
python -m pytest --cov=.

# Run specific test file
python -m pytest tests/test_feature.py
```

### Writing Tests

- Write tests for new functionality
- Include both unit and integration tests
- Test edge cases and error conditions
- Aim for good test coverage

## Submitting Changes

### 1. Create a Feature Branch

```bash
git checkout -b feature/your-feature-name
```

### 2. Make Your Changes

- Write clear, focused commits
- Use descriptive commit messages
- Follow the conventional commit format:
  ```
  type(scope): description
  
  [optional body]
  [optional footer]
  ```

Examples:
- `feat(model): add new ensemble method`
- `fix(data): correct preprocessing pipeline`
- `docs(readme): update installation instructions`

### 3. Test Your Changes

- Run the test suite
- Ensure all tests pass
- Test your changes manually if applicable

### 4. Update Documentation

- Update README.md if needed
- Add docstrings for new functions
- Update any relevant documentation

### 5. Submit a Pull Request

1. Push your branch to your fork
2. Create a pull request against the main branch
3. Fill out the pull request template
4. Request review from maintainers

## Issue Guidelines

### Before Creating an Issue

1. Check existing issues for duplicates
2. Search the documentation for solutions
3. Try to reproduce the issue

### Issue Template

When creating an issue, please include:

- **Title**: Clear, descriptive title
- **Description**: Detailed description of the issue
- **Steps to reproduce**: Step-by-step instructions
- **Expected behavior**: What you expected to happen
- **Actual behavior**: What actually happened
- **Environment**: OS, Python version, dependencies
- **Additional context**: Any other relevant information

### Issue Types

- **Bug**: Something isn't working
- **Feature**: Request for new functionality
- **Enhancement**: Improvement to existing functionality
- **Documentation**: Improvements to documentation
- **Question**: General questions about the project

## Pull Request Guidelines

### Before Submitting

- [ ] Code follows the style guidelines
- [ ] Tests pass and coverage is maintained
- [ ] Documentation is updated
- [ ] Commit messages are clear and descriptive
- [ ] Branch is up to date with main

### Pull Request Template

Your pull request should include:

- **Title**: Clear, descriptive title
- **Description**: Detailed description of changes
- **Type of change**: Bug fix, feature, documentation, etc.
- **Testing**: How you tested your changes
- **Breaking changes**: Any breaking changes
- **Related issues**: Links to related issues

### Review Process

1. Automated checks must pass
2. At least one maintainer must approve
3. All conversations must be resolved
4. Branch must be up to date with main

## Getting Help

- Check the documentation in the README
- Search existing issues and discussions
- Ask questions in the issues section
- Join community discussions

## Recognition

Contributors will be recognized in:
- The project README
- Release notes
- GitHub contributors list

Thank you for contributing to SmartAlert!