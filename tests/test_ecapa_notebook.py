import ast
import json
from pathlib import Path

import pytest


NOTEBOOK = Path(__file__).resolve().parents[1] / 'kaggle/ECAPA_Kaggle.ipynb'


def code_cells():
    notebook = json.loads(NOTEBOOK.read_text(encoding='utf-8'))
    return {cell['id']: ''.join(cell['source']) for cell in notebook['cells']
            if cell['cell_type'] == 'code'}


def test_notebook_syntax():
    for code in code_cells().values():
        ast.parse(code)
    # A literal backslash+n would put both package pins on a single line.
    tree = ast.parse(code_cells()['ecapa-03'])
    constants = [node.value for node in ast.walk(tree) if isinstance(node, ast.Constant)]
    assert '\n' in constants and '\\n' not in constants


def test_notebook_patch_is_idempotent_and_rejects_unknown_source(tmp_path):
    path = tmp_path / 'models/ecapa_tdnn.py'
    path.parent.mkdir()
    old = 'out = self.relu(self.bn1(self.conv1(x)))'
    new = 'out = self.bn1(self.relu(self.conv1(x)))'
    path.write_text(old, encoding='utf-8')
    code = code_cells()['ecapa-fix-block-order']
    for _ in range(2):
        exec(code, {'DST': tmp_path})
        assert path.read_text(encoding='utf-8') == new
    path.write_text('unknown implementation', encoding='utf-8')
    with pytest.raises(RuntimeError, match='Unknown encoder'):
        exec(code, {'DST': tmp_path})
    assert path.read_text(encoding='utf-8') == 'unknown implementation'
