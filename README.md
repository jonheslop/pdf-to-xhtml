# PDF to HTML

My helper script for Do book epub conversion

Usage from anywhere:

pdf-to-xhtml inspect /path/to/book.pdf
pdf-to-xhtml convert /path/to/book.pdf --out book.xhtml --body 10 --map 24=h1 --map 15=h2 --map 12.5=h3 --drop-below 9 --min-image-size 100 --small-caps-pattern '\b[A-Z]{4,}\b'

pdf-to-xhtml convert --help

Output files land in your current directory unless you pass an absolute --out path.

If you ever move the repo, just re-create the symlink:
ln -sf /new/path/pdf-to-xhtml ~/.local/bin/pdf-to-xhtml
