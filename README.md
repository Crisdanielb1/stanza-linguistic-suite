# stanza-linguistic-suite

**Extensible Stanza-based toolkit for linguistic analysis, corpus export, and visualization.**

This project provides a modular interface built on [Stanza](https://stanfordnlp.github.io/stanza/) for processing Spanish texts (including Costa Rican Spanish, which is the main purpose, but not limited to).  
It allows you to perform full linguistic annotation, export results in multiple formats, and generate visual summaries automatically.

---

## Features

- Linguistic analysis with Stanza (tokenization, POS tagging, lemmatization, dependency parsing, NER)
- Exports to TSV, CoNLL-U, Excel, and JSON formats
- Visualization tools for POS, lemmas, dependencies, and NER tags
- Automatic statistics (lemma frequencies, co-occurrences, sentence length, etc.)
- Persistent configuration (`config.json`)
- Command-line interface and interactive menu
- Special support for Spanish and Costa Rican Spanish corpora

---

## Project Structure

```
Stanza/
├── main.py               # Main entry point (CLI + menu)
├── menu.py               # Interactive text menu
├── modules/
│   ├── stanza_demo.py    # Core Stanza pipeline and helpers
│   ├── exporters.py      # TSV, CoNLL-U, Excel, JSON exporters
│   ├── plots.py          # Visualization utilities (matplotlib, wordcloud)
│   ├── stats.py          # Frequency and co-occurrence analysis
│   └── utils.py          # Config, I/O, and validation utilities
└── config.json           # Persistent settings
```

---

## Installation

Make sure you have **Python 3.9+** and Stanza:

pip install stanza pandas matplotlib openpyxl wordcloud

Then clone the repository:

git clone https://github.com/Crisdanielb1/stanza-linguistic-suite.git
cd stanza-linguistic-suite

Initialize Stanza models (example for Spanish):

import stanza
stanza.download('es')

---

## Usage

Run the tool interactively:

python -m Stanza.main

Or quickly process a text file to TSV:

python -m Stanza.main --input mi_texto.txt --quick-tsv

Results will be exported as `salida.tsv`, `salida.conllu`, `salida.xlsx`, or `salida.json` depending on your menu choices.

---

## Example Outputs

- Excel sheets summarizing lemma and POS frequencies  
- CoNLL-U annotated text for syntactic parsing  
- Visual plots of word frequencies, dependencies, and named entities  
- Word clouds highlighting the most frequent lemmas  

---

## Example Text

You can place your corpus or a sample text file such as:

mi_texto.txt

Example (Costa Rican Spanish):

“Diay mae, fuimos a la feria y compramos café y aguacates. Todo estaba carísimo, pero di, ni modo.”

---

## Contributing

Contributions and pull requests are welcome. You can:
- Add new visualization or export modules  
- Extend support for other languages  
- Improve the CLI or integrate with Jupyter notebooks  

---

## License

This project is licensed under the **MIT License**.

---

## Author

**Christopher Brenes Fernández**  
Indiana University – Hispanic Linguistics  
chbrenes@iu.edu
