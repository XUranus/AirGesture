## Build The Report

For author *AAAA*, *BBBB*, *CCCC*, and *DDDD*:

```bash
# substitude author placeholder
sed 's/Author_A/AAAA/g; s/Author_B/BBBB/g; s/Author_C/CCCC/g; s/Author_D/DDDD/g' report.tex > report.final.tex

# build pdf report
pdflatex report.final.tex
```
