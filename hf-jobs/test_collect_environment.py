#!/usr/bin/env python3

from __future__ import annotations

import json
import sys
import unittest
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from collect_environment import mac_hardware  # noqa: E402


class MacHardwareTests(unittest.TestCase):
    @patch("collect_environment.command")
    def test_device_identifiers_are_not_recorded(self, mocked_command) -> None:
        mocked_command.return_value = json.dumps({
            "SPHardwareDataType": [{
                "machine_name": "MacBook Pro",
                "chip_type": "Apple M4 Max",
                "serial_number": "secret",
                "platform_UUID": "also-secret",
            }]
        })
        result = mac_hardware()
        self.assertEqual(
            json.loads(result),
            {"machine_name": "MacBook Pro", "chip_type": "Apple M4 Max"},
        )
        self.assertNotIn("secret", result)


if __name__ == "__main__":
    unittest.main()
