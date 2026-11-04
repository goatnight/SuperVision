from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from spinevision_agent.rag import SQLiteVectorStore
from spinevision_agent.schemas import DiagnosisOutput, ReportOutput, SegmentationOutput
from spinevision_agent.tools.reporting import GenerateReportTool
from spinevision_agent.tools.retrieval import RetrieveKnowledgeTool
from spinevision_agent.tools.verification import VerifyCaseTool


class AgentToolTests(unittest.TestCase):
    def test_rag_report_and_verifier_keep_evidence_traceable(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            store = SQLiteVectorStore(root / "knowledge.sqlite3")
            seed = Path(__file__).resolve().parents[1] / "spinevision_agent" / "knowledge" / "seed_documents.json"
            retrieval = RetrieveKnowledgeTool(store, seed)
            retrieved = retrieval.invoke(
                {"query": "腰椎 MRI 低置信度人工复核和报告证据", "top_k": 3, "modality": "MRI"}
            )
            self.assertTrue(retrieved.data["evidence"])

            segmentation = SegmentationOutput(
                mask_path=root / "mask.nii.gz",
                input_shape=[64, 64, 8],
                output_shape=[64, 64, 8],
                labels_present=[1, 2],
                foreground_voxels=500,
                spacing=[1.0, 1.0, 3.0],
                qc_passed=True,
                label_semantics_validated=False,
            )
            diagnosis = DiagnosisOutput(
                predicted_class="spondylolisthesis_candidate",
                probabilities={"negative": 0.4, "spondylolisthesis_candidate": 0.6},
                confidence=0.6,
                uncertainty=0.4,
                model_name="test-classifier",
                requires_review=True,
            )
            report_result = GenerateReportTool().invoke(
                {
                    "modality": "MRI",
                    "segmentation": segmentation.model_dump(mode="json"),
                    "diagnosis": diagnosis.model_dump(mode="json"),
                    "evidence": retrieved.data["evidence"],
                }
            )
            report = ReportOutput.model_validate(report_result.data)
            verification = VerifyCaseTool().invoke(
                {
                    "segmentation": segmentation.model_dump(mode="json"),
                    "diagnosis": diagnosis.model_dump(mode="json"),
                    "report": report.model_dump(mode="json"),
                    "confidence_threshold": 0.7,
                }
            )
            codes = {item["code"] for item in verification.data["issues"]}
            self.assertIn("LABEL_SEMANTICS_UNVALIDATED", codes)
            self.assertIn("LOW_DIAGNOSIS_CONFIDENCE", codes)
            self.assertTrue(verification.data["human_review_required"])


if __name__ == "__main__":
    unittest.main()
