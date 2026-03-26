## Build The Report

For author *AAAA*, *BBBB*, *CCCC*, and *DDDD*:

```bash
cd report

# substitute author placeholders
sed 's/Author_A/AAAA/g; s/Author_B/BBBB/g; s/Author_C/CCCC/g; s/Author_D/DDDD/g' report.tex > report.final.tex

# build pdf report (run pdflatex + bibtex + pdflatex x2 for references)
pdflatex -interaction=nonstopmode report.final.tex
bibtex report.final
pdflatex -interaction=nonstopmode report.final.tex
pdflatex -interaction=nonstopmode report.final.tex
```

## Report Structure

```
report/
├── report.tex              # Main document (title, abstract, includes)
├── references.bib          # Bibliography database
├── sections/
│   ├── introduction.tex    # Section 1: Introduction & contributions
│   ├── related_work.tex    # Section 2: Related work
│   ├── data.tex            # Section 3: Data collection & preprocessing
│   ├── methodology.tex     # Section 4: TCN architecture & detection pipeline
│   ├── training.tex        # Section 5: Model training
│   ├── deployment.tex      # Section 6: Pruning, quantization, ONNX export
│   ├── application.tex     # Section 7: Android & Desktop apps, network protocol
│   ├── results.tex         # Section 8: Experimental results
│   ├── conclusion.tex      # Section 9: Conclusion & future work
│   └── contribution.tex    # Statement of contribution (mandatory)
├── figures/
│   ├── grab-release.png
│   ├── TCN-Architecture.png
│   ├── train-pipeline.png
│   ├── two-stage-pipeline.png
│   ├── hand.png
│   ├── confusion_matrix.png
│   └── training_curves.png
└── README.md               # This file
```

## Team Collaboration

Each section is in a separate `.tex` file under `sections/`. Team members can
edit their assigned sections independently:

| Member   | Files to edit                              |
|----------|-------------------------------------------|
| Author_A | `sections/data.tex`, `sections/training.tex` |
| Author_B | `sections/methodology.tex`, `sections/deployment.tex` |
| Author_C | `sections/application.tex` (Android part) |
| Author_D | `sections/application.tex` (Desktop part), `sections/results.tex` |
| Shared   | `sections/introduction.tex`, `sections/related_work.tex`, `sections/conclusion.tex` |
