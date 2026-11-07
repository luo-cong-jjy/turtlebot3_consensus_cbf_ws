from types import SimpleNamespace
import unittest
from unittest.mock import Mock
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from cbf_spawn_checked import ensure_spawned


class SpawnChecks(unittest.TestCase):
    def test_existing_model_is_not_silently_replaced_or_adopted(self):
        spawn=Mock()
        with self.assertRaises(RuntimeError):ensure_spawned(spawn,lambda:True,pause=Mock())
        spawn.assert_not_called()

    def test_timed_out_request_that_created_its_model_is_not_duplicated(self):
        spawn=Mock(return_value=SimpleNamespace(success=False,status_message='timeout'))
        self.assertEqual(ensure_spawned(spawn,Mock(side_effect=[False,True]),pause=Mock()),1)
        self.assertEqual(spawn.call_count,1)

    def test_retry_only_after_a_failed_request_left_the_model_missing(self):
        spawn=Mock(return_value=SimpleNamespace(success=True,status_message='ok'))
        exists=Mock(side_effect=[False]+[False]*10+[False,True])
        self.assertEqual(ensure_spawned(spawn,exists,pause=Mock()),2)
        self.assertEqual(spawn.call_count,2)

    def test_exhaustion_remains_a_failure(self):
        spawn=Mock(return_value=SimpleNamespace(success=False,status_message='failed'))
        with self.assertRaises(RuntimeError):ensure_spawned(spawn,lambda:False,pause=Mock())
        self.assertEqual(spawn.call_count,3)
