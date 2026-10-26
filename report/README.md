# Minor project report (ES-451)

Sources that generate `out/Minor_Project_Report.docx` (and a PDF rendering) following Annexure A of the department's notice: A4, margins 1.5 / 1.0 / 1.0 / 1.0 in, Times New Roman, 1.5 line spacing, Roman then Arabic page numbers, IEEE references.

```bash
cd report
npm install                      # docx
python make_diagrams.py          # diagrams into figures/ (needs matplotlib)
node build.js                    # builds the .docx, renders it with LibreOffice and fills the contents with real page numbers
```

- `content/` the chapters (plain JavaScript functions that return Word paragraphs and tables)
- `front.js` cover, declaration, certificate, acknowledgement, institute pages, abstract, contents and lists
- `refs.js` the IEEE reference list; `lib.js` numbering, captions, citations and layout helpers
- `figures/` diagrams (generated) and figures from the notebook run; `data/` result tables read by the content
- Highlighted yellow text in the document marks details that must be filled in (supervisor, principal, rubrics sheet).

`build.js` needs LibreOffice (`soffice`) and `pdftotext` on the path. It builds three times so that the table of contents and lists of figures and tables carry the real page numbers.
