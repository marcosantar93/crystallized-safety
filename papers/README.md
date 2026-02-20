# Papers

This directory intentionally contains a single paper aligned with the current canonical notebook:

- `DoseResponseSafety.tex`

## Build

From the repository root:

```bash
cd papers
pdflatex DoseResponseSafety.tex
pdflatex DoseResponseSafety.tex
```

If `dose_response_curve.png` is missing, first run
`notebooks/notebook_1_core_methodology.ipynb` to generate it, then copy/move the figure into
`papers/` and compile again.
