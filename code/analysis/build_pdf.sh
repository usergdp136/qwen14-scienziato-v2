#!/bin/bash
# Compila un articolo markdown (con formule $...$ / $$...$$) in PDF con pandoc + tectonic (LaTeX).
# Uso: build_pdf.sh input.md output.pdf   (le immagini sono cercate partendo dalla cartella del repo: --resource-path)
# Requisiti: pandoc >= 3, tectonic >= 0.15, font "DejaVu Serif" / "DejaVu Sans Mono".
set -e
IN="$1"; OUT="$2"; TMP=$(mktemp --suffix=.md)
# DejaVu non ha i glifi ✓/✗: li sostituisco con simboli matematici
sed -e 's/✓/$\\checkmark$/g' -e 's/✗/$\\times$/g' "$IN" > "$TMP"
pandoc "$TMP" -f markdown+tex_math_dollars+pipe_tables+yaml_metadata_block -o "$OUT" --pdf-engine=tectonic \
  --resource-path="$(dirname "$IN"):$(dirname "$IN")/..:." \
  -V mainfont="DejaVu Serif" -V monofont="DejaVu Sans Mono" -V fontsize=10pt -V geometry:margin=2.2cm \
  -V colorlinks=true -V linkcolor=blue -V urlcolor=blue --toc-depth=2 \
  -H <(printf '%s\n' '\usepackage{amsmath}' '\usepackage{amssymb}' '\usepackage{longtable}' '\usepackage{booktabs}' '\setlength{\emergencystretch}{3em}' '\sloppy')
rm -f "$TMP"
