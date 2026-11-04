from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import nibabel as nib
import numpy as np

from spinevision_agent.config import AgentSettings
from spinevision_agent.rag import SQLiteVectorStore
from spinevision_agent.schemas import CaseCreateRequest, RunStatus
from spinevision_agent.tools.classification import MockClassifierBackend, RunDiagnosisTool
from spinevision_agent.tools.imaging import InspectImageTool
from spinevision_agent.tools.registry import ToolRegistry
from spinevision_agent.tools.reporting import GenerateReportTool
from spinevision_agent.tools.retrieval import RetrieveKnowledgeTool
from spinevision_agent.tools.segmentation import MockSegmentationTool
from spinevision_agent.tools.verification import VerifyCaseTool
from spinevision_agent.workflow import SpineVisionWorkflow


class WorkflowTests(unittest.TestCase):
    def test_langgraph_workflow_routes_unvalidated_case_to_review(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            image_path = root / "case.nii.gz"
            nib.save(nib.Nifti1Image(np.ones((8, 8, 3), dtype=np.float32), np.eye(4)), image_path)
            settings = AgentSettings(
                artifact_root=root / "artifacts",
                knowledge_db=root / "knowledge.sqlite3",
                confidence_threshold=0.7,
            )
            registry = ToolRegistry()
            registry.register(InspectImageTool())
            registry.register(MockSegmentationTool())
            registry.register(RunDiagnosisTool(MockClassifierBackend(confidence=0.82), 0.7))
            seed = Path(__file__).resolve().parents[1] / "spinevision_agent" / "knowledge" / "seed_documents.json"
            registry.register(RetrieveKnowledgeTool(SQLiteVectorStore(settings.knowledge_db), seed))
            registry.register(GenerateReportTool())
            registry.register(VerifyCaseTool())
            workflow = SpineVisionWorkflow(settings, registry=registry)
            case = workflow.create_case(CaseCreateRequest(image_path=image_path))
            completed = workflow.run_case(case.case_id)

            self.assertEqual(completed.status, RunStatus.review)
            self.assertTrue(completed.human_review_required)
            self.assertIsNotNone(completed.report)
            self.assertEqual(
                [trace.tool_name for trace in completed.tool_trace],
                [
                    "inspect_mri_input",
                    "run_spine_segmentation",
                    "run_spine_diagnosis",
                    "retrieve_medical_knowledge",
                    "generate_grounded_report",
                    "verify_case_consistency",
                ],
            )


if __name__ == "__main__":
    unittest.main()
