import ast
import subprocess
import sys
from pathlib import Path

INPUTS_DIR = Path(__file__).parent / "inputs"


def main_loop(file: Path) -> None:
    file_ast = ast.parse(file.read_text(encoding="utf-8"), filename=str(file))
    # TODO: annotate
    source = ast.unparse(file_ast)
    result = subprocess.run(
        [sys.executable, "-m", "mypy", "--strict", "-c", source],
        capture_output=True,
        text=True,
    )
    if result.returncode == 0:
        print(f'======={str(file)}=========')
        print(source)


if __name__ == "__main__":
    for f in sorted(INPUTS_DIR.iterdir()):
        if f.is_file():
            main_loop(f)
