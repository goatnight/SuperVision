"use client";

import {
  Activity,
  AlertTriangle,
  BookOpen,
  BrainCircuit,
  Check,
  CheckCircle2,
  ChevronRight,
  CircleDot,
  FileCheck2,
  FileText,
  Gauge,
  ImageIcon,
  Layers3,
  ListChecks,
  LoaderCircle,
  PanelLeftClose,
  Play,
  Plus,
  RefreshCw,
  Server,
  Settings2,
  ShieldAlert,
  SlidersHorizontal,
  Sparkles,
  Upload,
  UserCheck,
  X,
  XCircle,
} from "lucide-react";
import { FormEvent, useCallback, useEffect, useRef, useState } from "react";

type RunStatus = "pending" | "running" | "partial" | "review" | "completed" | "failed";

const DEFAULT_API_BASE =
  process.env.NEXT_PUBLIC_SPINEVISION_API_URL ?? "http://127.0.0.1:8000";

type Evidence = {
  evidence_id: string;
  title: string;
  section: string;
  content: string;
  source: string;
  score: number;
};

type Finding = {
  finding_id: string;
  anatomical_level: string;
  observation: string;
  status: "observed" | "derived" | "inferred" | "unknown";
  confidence: number;
  source_tool: string;
  evidence_ids: string[];
};

type CaseRecord = {
  case_id: string;
  patient_hash: string;
  image_path: string;
  modality: string;
  user_goal: string;
  status: RunStatus;
  current_agent: string;
  segmentation?: {
    mask_path: string;
    output_shape: number[];
    labels_present: number[];
    foreground_voxels: number;
    spacing?: number[];
    qc_passed: boolean;
    qc_warnings: string[];
    label_semantics_validated: boolean;
  } | null;
  diagnosis?: {
    predicted_class: string;
    probabilities: Record<string, number>;
    confidence: number;
    uncertainty: number;
    model_name: string;
    requires_review: boolean;
  } | null;
  evidence: Evidence[];
  report?: {
    examination: string;
    findings: Finding[];
    impression: string[];
    evidence: Evidence[];
    limitations: string[];
    status: "draft" | "reviewed";
  } | null;
  verification?: {
    passed: boolean;
    human_review_required: boolean;
    issues: Array<{ code: string; severity: string; message: string; related_ids: string[] }>;
  } | null;
  tool_trace: Array<{
    step_id: string;
    agent: string;
    tool_name: string;
    result_status: string;
    started_at: string;
    finished_at: string;
    error_code?: string | null;
  }>;
  human_review_required: boolean;
  created_at: string;
  updated_at: string;
};

const demoTimestamp = Date.parse("2026-08-07T08:00:00.000Z");

const demoCase: CaseRecord = {
  case_id: "demo-lumbar-042",
  patient_hash: "8f4c2d91b6e8a042",
  image_path: "demo://lumbar-mri",
  modality: "MRI",
  user_goal: "腰椎 MRI 分割、滑脱候选识别与结构化报告",
  status: "review",
  current_agent: "HumanReview",
  segmentation: {
    mask_path: "demo://segmentation",
    output_shape: [384, 384, 16],
    labels_present: [1, 2, 3, 4, 5, 6],
    foreground_voxels: 42816,
    spacing: [0.68, 0.68, 4.0],
    qc_passed: true,
    qc_warnings: [],
    label_semantics_validated: false,
  },
  diagnosis: {
    predicted_class: "spondylolisthesis_candidate",
    probabilities: { no_spondylolisthesis: 0.18, spondylolisthesis_candidate: 0.82 },
    confidence: 0.82,
    uncertainty: 0.18,
    model_name: "DeepMR-PNP dual branch",
    requires_review: false,
  },
  evidence: [
    {
      evidence_id: "SOP-GROUNDING-001",
      title: "SpineVision 报告证据规范",
      section: "grounding",
      content: "报告中的节段、侧别和测量值必须来自结构化工具输出；证据不足时标记为未知。",
      source: "internal_sop",
      score: 0.91,
    },
    {
      evidence_id: "SOP-REVIEW-001",
      title: "SpineVision 人工复核规范",
      section: "human_review",
      content: "标签语义未验证、结果冲突或证据缺失时，病例不得自动完成，应进入人工复核。",
      source: "internal_sop",
      score: 0.87,
    },
  ],
  report: {
    examination: "腰椎 MRI",
    findings: [
      {
        finding_id: "F-SEG-001",
        anatomical_level: "unknown",
        observation: "分割工具检出 6 个非背景标签，前景体素数为 42816。",
        status: "observed",
        confidence: 1,
        source_tool: "run_spine_segmentation",
        evidence_ids: ["SOP-GROUNDING-001"],
      },
      {
        finding_id: "F-CLS-001",
        anatomical_level: "unknown",
        observation: "分类模型输出滑脱候选，置信度为 0.82。",
        status: "inferred",
        confidence: 0.82,
        source_tool: "run_spine_diagnosis",
        evidence_ids: ["SOP-GROUNDING-001", "SOP-REVIEW-001"],
      },
    ],
    impression: ["模型提示腰椎滑脱候选，仅作为待复核结果。"],
    evidence: [],
    limitations: ["分割标签的正式解剖语义尚未验证，未生成具体节段结论。", "本输出为研究用途报告草稿。"],
    status: "draft",
  },
  verification: {
    passed: false,
    human_review_required: true,
    issues: [
      {
        code: "LABEL_SEMANTICS_UNVALIDATED",
        severity: "high",
        message: "标签语义未验证，禁止生成具体腰椎节段结论。",
        related_ids: [],
      },
    ],
  },
  tool_trace: ["Planner", "Segmentation", "Diagnosis", "RAG", "Report", "Verifier"].map((agent, index) => ({
    step_id: `step-${String(index + 1).padStart(3, "0")}`,
    agent,
    tool_name: [
      "inspect_mri_input",
      "run_spine_segmentation",
      "run_spine_diagnosis",
      "retrieve_medical_knowledge",
      "generate_grounded_report",
      "verify_case_consistency",
    ][index],
    result_status: index === 5 ? "partial" : "success",
    started_at: new Date(demoTimestamp + index * 580).toISOString(),
    finished_at: new Date(demoTimestamp + (index + 1) * 580).toISOString(),
  })),
  human_review_required: true,
  created_at: new Date(demoTimestamp - 1000 * 60 * 18).toISOString(),
  updated_at: new Date(demoTimestamp + 6 * 580).toISOString(),
};

const statusLabels: Record<RunStatus, string> = {
  pending: "待处理",
  running: "分析中",
  partial: "部分完成",
  review: "待复核",
  completed: "已完成",
  failed: "已失败",
};

const tabs = [
  { id: "report", label: "报告", icon: FileText },
  { id: "evidence", label: "证据", icon: BookOpen },
  { id: "trace", label: "轨迹", icon: ListChecks },
  { id: "benchmark", label: "评测", icon: Gauge },
] as const;

type TabId = (typeof tabs)[number]["id"];

function DemoMRI({ overlay, opacity }: { overlay: boolean; opacity: number }) {
  const canvasRef = useRef<HTMLCanvasElement>(null);

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const ctx = canvas.getContext("2d");
    if (!ctx) return;
    const width = 560;
    const height = 620;
    const image = ctx.createImageData(width, height);
    for (let y = 0; y < height; y += 1) {
      for (let x = 0; x < width; x += 1) {
        const dx = (x - width * 0.48) / width;
        const dy = (y - height * 0.5) / height;
        const body = Math.max(0, 1 - (dx * dx * 5.2 + dy * dy * 2.6));
        const texture = Math.sin(x * 0.07) * 4 + Math.cos(y * 0.04) * 3;
        const value = Math.max(8, Math.min(188, 18 + body * 118 + texture));
        const offset = (y * width + x) * 4;
        image.data[offset] = value;
        image.data[offset + 1] = value;
        image.data[offset + 2] = value + 2;
        image.data[offset + 3] = 255;
      }
    }
    ctx.putImageData(image, 0, 0);
    ctx.fillStyle = "rgba(16, 19, 22, 0.86)";
    ctx.fillRect(0, 0, 80, height);
    ctx.fillRect(width - 60, 0, 60, height);
    const vertebrae = [128, 205, 282, 359, 436, 513];
    vertebrae.forEach((cy, index) => {
      ctx.save();
      ctx.translate(300 + index * 3, cy);
      ctx.rotate(-0.055 + index * 0.012);
      ctx.fillStyle = index === 5 ? "rgba(202, 202, 194, 0.92)" : "rgba(224, 224, 215, 0.94)";
      ctx.fillRect(-74, -25, 118, 51);
      ctx.fillStyle = "rgba(92, 94, 91, 0.88)";
      ctx.fillRect(-62, 27, 94, 12);
      if (overlay) {
        const colors = ["48,196,141", "55,149,224", "234,179,63", "221,89,95", "140,108,212", "61,188,196"];
        ctx.strokeStyle = `rgba(${colors[index]},${opacity})`;
        ctx.lineWidth = 8;
        ctx.strokeRect(-76, -27, 122, 55);
      }
      ctx.restore();
    });
    ctx.fillStyle = "rgba(236, 239, 235, 0.58)";
    ctx.beginPath();
    ctx.ellipse(330, 316, 21, 238, -0.03, 0, Math.PI * 2);
    ctx.fill();
  }, [overlay, opacity]);

  return <canvas ref={canvasRef} width={560} height={620} aria-label="腰椎 MRI 演示影像" />;
}

function durationMs(start: string, end: string) {
  return Math.max(0, new Date(end).getTime() - new Date(start).getTime());
}

export function Workbench() {
  const [apiBase, setApiBase] = useState(DEFAULT_API_BASE);
  const [cases, setCases] = useState<CaseRecord[]>([demoCase]);
  const [selectedId, setSelectedId] = useState(demoCase.case_id);
  const [connected, setConnected] = useState(false);
  const [busy, setBusy] = useState(false);
  const [activeTab, setActiveTab] = useState<TabId>("report");
  const [overlay, setOverlay] = useState(true);
  const [opacity, setOpacity] = useState(0.5);
  const [slice, setSlice] = useState(7);
  const [uploadOpen, setUploadOpen] = useState(false);
  const [editOpen, setEditOpen] = useState(false);
  const [draftImpression, setDraftImpression] = useState("");
  const [toast, setToast] = useState("");
  const selected = cases.find((item) => item.case_id === selectedId) ?? demoCase;
  const isDemo = selected.case_id === demoCase.case_id;

  const notify = (message: string) => {
    setToast(message);
    window.setTimeout(() => setToast(""), 2600);
  };

  const loadCases = useCallback(async () => {
    try {
      const response = await fetch(`${apiBase}/cases`, { cache: "no-store" });
      if (!response.ok) throw new Error("API unavailable");
      const remoteCases = (await response.json()) as CaseRecord[];
      setConnected(true);
      setCases([demoCase, ...remoteCases.sort((a, b) => b.updated_at.localeCompare(a.updated_at))]);
    } catch {
      setConnected(false);
      setCases((current) => (current.some((item) => item.case_id === demoCase.case_id) ? current : [demoCase, ...current]));
    }
  }, [apiBase]);

  useEffect(() => {
    const timer = window.setTimeout(() => void loadCases(), 0);
    return () => window.clearTimeout(timer);
  }, [loadCases]);

  useEffect(() => {
    const timer = window.setTimeout(() => {
      const count = selected.segmentation?.output_shape?.[2] ?? 16;
      setSlice(Math.min(Math.floor(count / 2), Math.max(0, count - 1)));
      setDraftImpression(selected.report?.impression.join("\n") ?? "");
    }, 0);
    return () => window.clearTimeout(timer);
  }, [selectedId, selected.segmentation?.output_shape, selected.report?.impression]);

  const uploadCase = async (event: FormEvent<HTMLFormElement>) => {
    event.preventDefault();
    const form = event.currentTarget;
    const data = new FormData(form);
    setBusy(true);
    try {
      const response = await fetch(`${apiBase}/cases/upload`, { method: "POST", body: data });
      if (!response.ok) throw new Error(await response.text());
      const created = (await response.json()) as CaseRecord;
      await loadCases();
      setSelectedId(created.case_id);
      setUploadOpen(false);
      notify("病例已进入队列");
    } catch (error) {
      notify(error instanceof Error ? error.message : "病例上传失败");
    } finally {
      setBusy(false);
    }
  };

  const runSelected = async () => {
    if (isDemo) {
      notify("演示病例已完成分析");
      return;
    }
    setBusy(true);
    try {
      const response = await fetch(`${apiBase}/cases/${selected.case_id}/run`, { method: "POST" });
      if (!response.ok) throw new Error(await response.text());
      const updated = (await response.json()) as CaseRecord;
      setCases((current) => current.map((item) => (item.case_id === updated.case_id ? updated : item)));
      notify(updated.human_review_required ? "分析完成，等待人工复核" : "分析完成");
    } catch (error) {
      notify(error instanceof Error ? error.message : "分析执行失败");
    } finally {
      setBusy(false);
    }
  };

  const review = async (decision: "accept" | "modify" | "reject") => {
    if (isDemo) {
      setCases((current) => current.map((item) => item.case_id === selected.case_id
        ? { ...item, status: decision === "reject" ? "failed" : "completed", human_review_required: decision === "reject" }
        : item));
      setEditOpen(false);
      notify(decision === "reject" ? "演示报告已驳回" : "演示报告已确认");
      return;
    }
    const correctedReport = decision === "modify" && selected.report
      ? { ...selected.report, impression: draftImpression.split("\n").filter(Boolean) }
      : undefined;
    const response = await fetch(`${apiBase}/cases/${selected.case_id}/review`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        decision,
        reviewer: "workbench_user",
        reason: decision === "accept" ? "人工复核通过" : decision === "modify" ? "人工修订报告" : "人工复核驳回",
        corrected_report: correctedReport,
      }),
    });
    if (!response.ok) {
      notify("复核提交失败");
      return;
    }
    const updated = (await response.json()) as CaseRecord;
    setCases((current) => current.map((item) => (item.case_id === updated.case_id ? updated : item)));
    setEditOpen(false);
    notify("复核结果已保存");
  };

  const maxSlice = Math.max(0, (selected.segmentation?.output_shape?.[2] ?? 16) - 1);
  const confidence = selected.diagnosis?.confidence ?? 0;
  const previewUrl = `${apiBase}/cases/${selected.case_id}/preview?slice_index=${slice}&overlay=${overlay}&opacity=${opacity}`;

  return (
    <main className="app-shell">
      <aside className="sidebar">
        <div className="brand-row">
          <div className="brand-mark"><Activity size={20} /></div>
          <div><strong>SpineVision</strong><span>Agent Workbench</span></div>
          <button className="icon-button mobile-only" title="收起侧栏" aria-label="收起侧栏"><PanelLeftClose size={18} /></button>
        </div>

        <button className="new-case-button" onClick={() => setUploadOpen(true)}>
          <Plus size={17} /> 新建病例
        </button>

        <div className="queue-heading">
          <span>病例队列</span>
          <button className="icon-button" onClick={loadCases} title="刷新病例" aria-label="刷新病例"><RefreshCw size={15} /></button>
        </div>
        <div className="case-list">
          {cases.map((item) => (
            <button
              key={item.case_id}
              className={`case-row ${selectedId === item.case_id ? "active" : ""}`}
              onClick={() => setSelectedId(item.case_id)}
            >
              <span className={`status-dot ${item.status}`} />
              <span className="case-copy">
                <strong>{item.case_id === demoCase.case_id ? "演示病例 · 042" : item.case_id.replace("case-", "病例 ")}</strong>
                <small>{item.modality} · {statusLabels[item.status]}</small>
              </span>
              <ChevronRight size={15} />
            </button>
          ))}
        </div>

        <div className="sidebar-foot">
          <div className="connection-row">
            <Server size={15} />
            <span>{connected ? "Agent API 已连接" : "离线演示模式"}</span>
            <i className={connected ? "online" : "offline"} />
          </div>
          <label className="api-field">
            <span><Settings2 size={14} /> API 地址</span>
            <input value={apiBase} onChange={(event) => setApiBase(event.target.value)} />
          </label>
        </div>
      </aside>

      <section className="workspace">
        <header className="workspace-header">
          <div>
            <div className="eyebrow-row">
              <span>{isDemo ? "DEMO CASE" : selected.patient_hash}</span>
              <span className={`status-badge ${selected.status}`}>{statusLabels[selected.status]}</span>
            </div>
            <h1>{isDemo ? "腰椎 MRI 智能分析" : selected.case_id}</h1>
          </div>
          <div className="header-actions">
            <button className="secondary-button" onClick={loadCases}><RefreshCw size={16} />刷新</button>
            <button className="primary-button" onClick={runSelected} disabled={busy || selected.status === "running"}>
              {busy ? <LoaderCircle className="spin" size={17} /> : <Play size={17} />}
              {selected.status === "pending" ? "运行分析" : "重新运行"}
            </button>
          </div>
        </header>

        <div className="research-banner">
          <ShieldAlert size={16} />
          <span>研究用途 · 自动结果为候选和报告草稿，最终结论须由专业人员复核</span>
        </div>

        <div className="work-grid">
          <section className="viewer-section">
            <div className="section-heading">
              <div><ImageIcon size={17} /><strong>影像阅览</strong><span>矢状位 · Slice {slice + 1}/{maxSlice + 1}</span></div>
              <div className="viewer-tools">
                <label className="toggle-label">
                  <input type="checkbox" checked={overlay} onChange={(event) => setOverlay(event.target.checked)} />
                  <span /> Mask
                </label>
                <label className="opacity-control" title="Mask 透明度">
                  <SlidersHorizontal size={15} />
                  <input type="range" min="0" max="1" step="0.05" value={opacity} onChange={(event) => setOpacity(Number(event.target.value))} />
                </label>
              </div>
            </div>
            <div className="image-stage">
              {isDemo ? <DemoMRI overlay={overlay} opacity={opacity} /> : (
                <img src={previewUrl} alt="当前病例腰椎 MRI 预览" key={previewUrl} />
              )}
              <span className="orientation-marker top">S</span>
              <span className="orientation-marker bottom">I</span>
              <span className="image-watermark">RESEARCH PREVIEW</span>
            </div>
            <div className="slice-control">
              <span>1</span>
              <input type="range" min="0" max={maxSlice} value={slice} onChange={(event) => setSlice(Number(event.target.value))} />
              <span>{maxSlice + 1}</span>
            </div>

            <div className="agent-rail">
              {selected.tool_trace.length ? selected.tool_trace.map((trace) => (
                <div className={`agent-step ${trace.result_status}`} key={trace.step_id}>
                  <span>{trace.result_status === "success" ? <Check size={13} /> : <AlertTriangle size={13} />}</span>
                  <div><strong>{trace.agent}</strong><small>{durationMs(trace.started_at, trace.finished_at)} ms</small></div>
                </div>
              )) : ["Planner", "Segmentation", "Diagnosis", "RAG", "Report", "Verifier"].map((agent) => (
                <div className="agent-step pending" key={agent}><span><CircleDot size={13} /></span><div><strong>{agent}</strong><small>等待</small></div></div>
              ))}
            </div>
          </section>

          <aside className="result-section">
            <div className="section-heading"><div><BrainCircuit size={17} /><strong>分析摘要</strong></div></div>

            <div className="result-block diagnosis-block">
              <div className="block-label"><span>病变候选</span><Sparkles size={15} /></div>
              <strong className="diagnosis-name">{selected.diagnosis?.predicted_class ?? "等待模型分析"}</strong>
              <div className="confidence-row"><span>置信度</span><strong>{Math.round(confidence * 100)}%</strong></div>
              <div className="confidence-track"><span style={{ width: `${confidence * 100}%` }} /></div>
              <small>{selected.diagnosis?.model_name ?? "Diagnosis Agent 尚未执行"}</small>
            </div>

            <div className="result-block">
              <div className="block-label"><span>分割质量</span><Layers3 size={15} /></div>
              <div className="metric-grid">
                <div><strong>{selected.segmentation?.labels_present.length ?? "–"}</strong><span>标签数</span></div>
                <div><strong>{selected.segmentation?.foreground_voxels.toLocaleString() ?? "–"}</strong><span>前景体素</span></div>
                <div><strong>{selected.segmentation?.qc_passed ? "通过" : "待检"}</strong><span>Mask QC</span></div>
              </div>
            </div>

            <div className="result-block verifier-block">
              <div className="block-label">
                <span>Verifier</span>
                {selected.verification?.passed ? <CheckCircle2 size={16} /> : <AlertTriangle size={16} />}
              </div>
              {selected.verification?.issues?.length ? selected.verification.issues.map((issue) => (
                <div className={`issue-row ${issue.severity}`} key={issue.code}>
                  <AlertTriangle size={15} />
                  <div><strong>{issue.code}</strong><span>{issue.message}</span></div>
                </div>
              )) : (
                <div className="empty-check"><CheckCircle2 size={17} />未发现结构化冲突</div>
              )}
            </div>

            <div className="review-actions">
              <button className="reject-button" onClick={() => review("reject")}><XCircle size={17} />驳回</button>
              <button className="secondary-button" onClick={() => setEditOpen(true)}><FileText size={17} />修订</button>
              <button className="accept-button" onClick={() => review("accept")}><UserCheck size={17} />确认</button>
            </div>
          </aside>
        </div>

        <section className="detail-section">
          <div className="tab-list" role="tablist">
            {tabs.map(({ id, label, icon: Icon }) => (
              <button key={id} className={activeTab === id ? "active" : ""} onClick={() => setActiveTab(id)} role="tab">
                <Icon size={16} />{label}
              </button>
            ))}
          </div>
          <div className="tab-content">
            {activeTab === "report" && (
              <div className="report-layout">
                <div>
                  <h2>影像所见</h2>
                  {selected.report?.findings.map((finding) => (
                    <div className="finding-row" key={finding.finding_id}>
                      <span className={`finding-type ${finding.status}`}>{finding.status}</span>
                      <p>{finding.observation}</p>
                      <small>{finding.source_tool} · {Math.round(finding.confidence * 100)}%</small>
                    </div>
                  )) ?? <p className="empty-copy">尚未生成报告。</p>}
                </div>
                <div className="impression-band">
                  <h2>印象</h2>
                  {selected.report?.impression.map((item) => <p key={item}>{item}</p>)}
                  {selected.report?.limitations.map((item) => <div className="limitation" key={item}><AlertTriangle size={14} />{item}</div>)}
                </div>
              </div>
            )}
            {activeTab === "evidence" && (
              <div className="evidence-grid">
                {selected.evidence.length ? selected.evidence.map((item) => (
                  <article className="evidence-card" key={item.evidence_id}>
                    <div><BookOpen size={16} /><span>{item.evidence_id}</span><strong>{Math.round(item.score * 100)}%</strong></div>
                    <h3>{item.title}</h3><p>{item.content}</p><small>{item.source} · {item.section}</small>
                  </article>
                )) : <p className="empty-copy">尚未检索到证据。</p>}
              </div>
            )}
            {activeTab === "trace" && (
              <div className="trace-table">
                <div className="trace-head"><span>步骤</span><span>Agent / Tool</span><span>结果</span><span>耗时</span></div>
                {selected.tool_trace.map((trace) => (
                  <div className="trace-row" key={trace.step_id}>
                    <span>{trace.step_id}</span><span><strong>{trace.agent}</strong><small>{trace.tool_name}</small></span>
                    <span className={`trace-status ${trace.result_status}`}>{trace.result_status}</span>
                    <span>{durationMs(trace.started_at, trace.finished_at)} ms</span>
                  </div>
                ))}
              </div>
            )}
            {activeTab === "benchmark" && (
              <div className="benchmark-grid">
                {[
                  ["6.0", "Tool / 病例", "正常流程"], ["2.02 s", "平均耗时", "真实分割 · CPU"],
                  ["87.5%", "RAG Recall@3", "8 个标注查询"], ["100%", "字段契约", "100 个合成报告"],
                  ["100%", "冲突检出", "120 个注入错误"], ["100%", "异常恢复", "50 个瞬时故障"],
                ].map(([value, label, note]) => (
                  <div key={label}><strong>{value}</strong><span>{label}</span><small>{note}</small></div>
                ))}
              </div>
            )}
          </div>
        </section>
      </section>

      {uploadOpen && (
        <div className="modal-backdrop" role="dialog" aria-modal="true" aria-label="导入腰椎 MRI">
          <form className="modal" onSubmit={uploadCase}>
            <div className="modal-header"><div><Upload size={19} /><strong>导入腰椎 MRI</strong></div><button type="button" className="icon-button" onClick={() => setUploadOpen(false)}><X size={18} /></button></div>
            <label className="upload-zone">
              <Upload size={28} /><strong>选择 NIfTI 文件</strong><span>.nii 或 .nii.gz，最大 512 MB</span>
              <input type="file" name="file" accept=".nii,.nii.gz" required />
            </label>
            <label className="form-field"><span>分析目标</span><textarea name="user_goal" defaultValue="完成腰椎 MRI 分割、病变候选识别及结构化报告草稿" /></label>
            <div className="modal-actions"><button type="button" className="secondary-button" onClick={() => setUploadOpen(false)}>取消</button><button className="primary-button" disabled={busy}>{busy ? <LoaderCircle className="spin" size={17} /> : <Upload size={17} />}导入病例</button></div>
          </form>
        </div>
      )}

      {editOpen && selected.report && (
        <div className="modal-backdrop" role="dialog" aria-modal="true" aria-label="修订报告印象">
          <div className="modal report-modal">
            <div className="modal-header"><div><FileCheck2 size={19} /><strong>修订报告印象</strong></div><button className="icon-button" onClick={() => setEditOpen(false)}><X size={18} /></button></div>
            <label className="form-field"><span>印象</span><textarea value={draftImpression} onChange={(event) => setDraftImpression(event.target.value)} rows={7} /></label>
            <div className="modal-actions"><button className="secondary-button" onClick={() => setEditOpen(false)}>取消</button><button className="accept-button" onClick={() => review("modify")}><Check size={17} />保存并确认</button></div>
          </div>
        </div>
      )}

      {toast && <div className="toast"><CheckCircle2 size={17} />{toast}</div>}
    </main>
  );
}
