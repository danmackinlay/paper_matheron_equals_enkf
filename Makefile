# Convenience Makefile for doit automation
# This is a simple wrapper around dodo.py for users who prefer make syntax

.PHONY: pdf figures timing_data test clean help

# Default target
pdf:
	uv run doit pdf

# Individual targets  
figures:
	uv run doit figures

timing_data:
	uv run doit timing_data

test:
	uv run doit test

# Clean targets
clean:
	uv run doit clean

clean-data:
	uv run doit clean_data

clean-figures:
	uv run doit clean_figures

clean-latex:
	uv run doit clean_latex

# Help
help:
	@echo "Available targets:"
	@echo "  pdf          - Build complete pipeline: benchmarks → figures → main.pdf"
	@echo "  figures      - Generate all figures"
	@echo "  timing_data  - Generate timing CSV data"
	@echo "  test         - Run test suite"
	@echo "  clean        - Clean all generated files"
	@echo "  clean-data   - Clean CSV files"
	@echo "  clean-figures- Clean figure PDFs"
	@echo "  clean-latex  - Clean LaTeX auxiliary files"
	@echo "  help         - Show this help"
	@echo ""
	@echo "For more detailed control, use 'doit' directly:"
	@echo "  uv run doit list     # Show all available tasks"
	@echo "  uv run doit -n 4 pdf # Run with 4 parallel processes"