"""把 percent 格式的 conv_lab.py 转成并执行 Jupyter Notebook（notebook.ipynb）。

用法：`python build_notebook.py`
conv_lab.py 是本仓库的唯一源码，本脚本只是把它拆成 notebook 单元格并执行，
保证「脚本」和「notebook」两份产物内容一致。
"""
from __future__ import annotations

import re
from pathlib import Path

import nbformat
from nbconvert.preprocessors import ExecutePreprocessor

ROOT = Path(__file__).resolve().parent
SRC = ROOT / "conv_lab.py"
OUT = ROOT / "notebook.ipynb"


def parse_percent(text: str):
    """解析 jupytext percent 格式，返回 [(kind, source), ...]。"""
    cells = []
    kind = None
    buf: list[str] = []

    def flush():
        nonlocal kind, buf
        if kind is None:
            return
        if kind == "markdown":
            src = "\n".join(re.sub(r"^#\s?", "", ln) for ln in buf).strip("\n")
        else:
            src = "\n".join(buf).strip("\n")
        if src.strip():
            cells.append((kind, src))
        buf = []

    for line in text.splitlines():
        m = re.match(r"^#\s*%%(.*)$", line)
        if m:
            flush()
            kind = "markdown" if "[markdown]" in m.group(1) else "code"
            continue
        if kind is None:
            continue
        buf.append(line)
    flush()
    return cells


def main() -> None:
    cells = parse_percent(SRC.read_text(encoding="utf-8"))

    nb = nbformat.v4.new_notebook()
    for kind, src in cells:
        if kind == "markdown":
            nb.cells.append(nbformat.v4.new_markdown_cell(src))
        else:
            nb.cells.append(nbformat.v4.new_code_cell(src))
    nb.metadata["kernelspec"] = {
        "name": "python3",
        "display_name": "Python 3",
        "language": "python",
    }
    nb.metadata["language_info"] = {"name": "python", "pygments_lexer": "ipython3"}

    print(f"解析出 {len(nb.cells)} 个单元格，开始执行 ...")
    ep = ExecutePreprocessor(timeout=900, kernel_name="python3")
    ep.preprocess(nb, {"metadata": {"path": str(ROOT)}})

    nbformat.write(nb, OUT)
    n_code = sum(1 for c in nb.cells if c.cell_type == "code")
    print(f"已写出 {OUT.name}（{len(nb.cells)} 个单元格，其中代码 {n_code} 个，含执行结果）")


if __name__ == "__main__":
    main()
