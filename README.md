```markdown
# Knight_Tour_Problem

This repository contains a small Python package that models a chessboard as a
graph where each vertex is a square and edges represent legal knight moves.

Files added in this change:

- `knight_graph/board.py` – BoardGraph implementation producing an adjacency
	list for knight moves.
- `examples/run.py` – small runner demonstrating usage.
- `tests/test_board.py` – unit tests (uses Python's builtin `unittest`).

Quick start
-----------

Run the example:

```powershell
python .\examples\run.py
```

Run the tests:

```powershell
python -m unittest discover -v
```

Implementation notes
--------------------

The main type is `BoardGraph(rows, cols)` which exposes:

- `rc_to_index(r, c)` and `index_to_rc(idx)` for conversions between 2D
	coordinates and a single integer index.
- `adj` — a dict mapping each index to a list of neighbor indices (knight moves).
- `neighbors(pos)` — accepts an index or an `(r, c)` tuple and returns a list
	of neighbor indices.

The default board is 8x8.

License
-------
MIT-style (add your license here).

```