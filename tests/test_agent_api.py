from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import nibabel as nib
import numpy as np
from fastapi.testclient import TestClient

from spinevision_agent.api import create_app
from spinevision_agent.config import AgentSettings
from spinevision_agent.workflow import SpineVisionWorkflow


class AgentApiTests(unittest.TestCase):
    def test_upload_list_and_preview_nifti_case(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "source.nii.gz"
            volume = np.zeros((12, 10, 5), dtype=np.float32)
            volume[3:9, 2:8, :] = np.linspace(0.1, 1.0, 6, dtype=np.float32)[:, None, None]
            nib.save(nib.Nifti1Image(volume, np.eye(4)), source)
            settings = AgentSettings(
                artifact_root=root / "artifacts",
                knowledge_db=root / "knowledge.sqlite3",
                segmentation_checkpoint=root / "missing.pt",
                classifier_backend="mock",
            )
            client = TestClient(create_app(SpineVisionWorkflow(settings)))

            with source.open("rb") as stream:
                response = client.post(
                    "/cases/upload",
                    files={"file": ("lumbar.nii.gz", stream, "application/gzip")},
                    data={"user_goal": "research preview", "patient_hash": "anonymous-case"},
                )

            self.assertEqual(response.status_code, 200)
            case = response.json()
            self.assertEqual(case["status"], "pending")
            self.assertNotIn("anonymous-case", case["patient_hash"])

            listed = client.get("/cases")
            self.assertEqual(listed.status_code, 200)
            self.assertEqual(len(listed.json()), 1)

            preview = client.get(f"/cases/{case['case_id']}/preview", params={"slice_index": 2})
            self.assertEqual(preview.status_code, 200)
            self.assertEqual(preview.headers["content-type"], "image/png")
            self.assertEqual(preview.headers["x-slice-index"], "2")
            self.assertEqual(preview.headers["x-slice-count"], "5")
            self.assertTrue(preview.content.startswith(b"\x89PNG"))

    def test_upload_rejects_non_nifti_file(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            settings = AgentSettings(
                artifact_root=root / "artifacts",
                knowledge_db=root / "knowledge.sqlite3",
                segmentation_checkpoint=root / "missing.pt",
                classifier_backend="mock",
            )
            client = TestClient(create_app(SpineVisionWorkflow(settings)))
            response = client.post(
                "/cases/upload",
                files={"file": ("notes.txt", b"not an image", "text/plain")},
            )
            self.assertEqual(response.status_code, 422)


if __name__ == "__main__":
    unittest.main()
