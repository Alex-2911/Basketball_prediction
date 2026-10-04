import importlib.util
import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "normalize_bbr_schedule_snapshot.py"
SPEC = importlib.util.spec_from_file_location("bbr_schedule_snapshot", SCRIPT)
normalizer = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = normalizer
SPEC.loader.exec_module(normalizer)


class BbrScheduleSnapshotTests(unittest.TestCase):
    def test_ed_time_converts_to_utc_during_dst(self):
        out = normalizer.parse_bbr_datetime("Tue, Oct 20, 2026", "3:00p")
        self.assertEqual(out.strftime("%Y-%m-%dT%H:%M:%SZ"), "2026-10-20T19:00:00Z")

    def test_et_time_converts_to_utc_after_dst(self):
        out = normalizer.parse_bbr_datetime("Mon, Jan 4, 2027", "7:30p")
        self.assertEqual(out.strftime("%Y-%m-%dT%H:%M:%SZ"), "2027-01-05T00:30:00Z")


if __name__ == "__main__":
    unittest.main()
