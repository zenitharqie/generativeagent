# Replay sample data

This directory intentionally contains no generated research or pilot output.

Create a replay JSON outside any source run or overlay directory with:

```text
python3 main.py export-visualizer-data \
  --run-dir <canonical-run-directory> \
  --output <fresh-output-file.json>
```

Then open `visualizer/index.html` and select that local JSON file. The browser
uses `FileReader`; it does not upload the file or contact a backend.
