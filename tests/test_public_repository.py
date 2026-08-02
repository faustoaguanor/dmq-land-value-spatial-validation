from __future__ import annotations

import csv
import re
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


class PublicRepositoryContracts(unittest.TestCase):
    def test_restricted_file_types_are_absent(self):
        forbidden = {".gpkg", ".shp", ".shx", ".dbf", ".tif", ".tiff", ".pt", ".pth", ".onnx", ".zip", ".docx"}
        found = [p.relative_to(ROOT).as_posix() for p in ROOT.rglob("*") if p.is_file() and ".git" not in p.parts and p.suffix.lower() in forbidden]
        self.assertEqual(found, [])

    def test_restricted_names_are_absent(self):
        patterns = ("dataset.csv", "dataset.gpkg", "val_catas.csv", "fold_assignments.csv", "split.csv")
        found = [p.relative_to(ROOT).as_posix() for p in ROOT.rglob("*") if p.is_file() and p.name.lower() in patterns]
        self.assertEqual(found, [])

    def test_no_absolute_personal_paths_in_text(self):
        suffixes = {".py", ".md", ".txt", ".yml", ".yaml", ".toml"}
        bad = []
        for path in ROOT.rglob("*"):
            if not path.is_file() or ".git" in path.parts or path.suffix.lower() not in suffixes:
                continue
            text = path.read_text(encoding="utf-8", errors="ignore")
            if re.search(r"C:[\\/]Users[\\/]", text, flags=re.IGNORECASE):
                bad.append(path.relative_to(ROOT).as_posix())
        self.assertEqual(bad, [])

    def test_published_tables_have_no_record_identifiers(self):
        forbidden_columns = {"predio", "predio_join", "x", "y", "geometry", "lat", "lon", "longitude", "latitude"}
        for path in ROOT.glob("results/**/*.csv"):
            with path.open(encoding="utf-8", newline="") as handle:
                header = {column.strip().lower() for column in next(csv.reader(handle))}
            self.assertFalse(header & forbidden_columns, path.name)

    def test_holdout_ranking_matches_thesis(self):
        with (ROOT / "results" / "holdout.csv").open(encoding="utf-8", newline="") as handle:
            rows = list(csv.DictReader(handle))
        self.assertEqual([row["model"] for row in rows], ["Random Forest", "SANNWR", "GWR", "GNNWR", "OLS"])
        self.assertAlmostEqual(float(rows[0]["rmse"]), 76.99)
        self.assertAlmostEqual(float(rows[-1]["rmse"]), 139.01)


if __name__ == "__main__":
    unittest.main()
