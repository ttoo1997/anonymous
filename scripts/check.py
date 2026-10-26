"""Verify source integrity and run the bundled unit tests."""
import unittest
from pathlib import Path
from paper_suite.frozen import verify


def main():
    root=Path(__file__).resolve().parents[1]
    print('Algorithm version:',verify())
    suite=unittest.TestSuite()
    suite.addTests(unittest.defaultTestLoader.discover(str(root/'paper_suite/tests'),top_level_dir=str(root)))
    suite.addTests(unittest.defaultTestLoader.discover(str(root/'tests'),top_level_dir=str(root)))
    result=unittest.TextTestRunner(verbosity=1).run(suite)
    raise SystemExit(0 if result.wasSuccessful() else 1)


if __name__=='__main__':
    main()
