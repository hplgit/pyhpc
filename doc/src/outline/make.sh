#!/bin/sh
name=project

doconce format html ${name} --html_style=bootswatch_journal
doconce split_html ${name}

doconce format pdflatex ${name}
doconce ptex2tex ${name}
pdflatex ${name}

cp ${name}.html ._*.html ${name}.pdf ../../pub
