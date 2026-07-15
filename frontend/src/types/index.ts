// ==================== API 请求/响应类型 ====================

export interface ChatRequest {
  question: string;
  session_id?: string | null;
  user_id: string;
  mode?: string;  // "auto" | "react" | "plan" | "multi_agent"
}

export interface ChatResponse {
  answer: string;
  session_id: string;
  turn_count: number;
  mode: string;
  verification: VerificationInfo;
  image_desc?: string;
  detected_products?: string[];
}

export interface VerificationInfo {
  pass?: boolean;
  score?: number;
  reason?: string;
  suggestion?: string;
}

export interface SessionSummary {
  session_id: string;
  user_id: string;
  turn_count: number;
  created_at: string;
}

export interface MessageRecord {
  role: string;
  content: string;
  image_count?: number;
}

export interface SessionHistory {
  session_id: string;
  messages: MessageRecord[];
  summary: string;
}

export interface UserProfile {
  user_id: string;
  data: {
    profile: Record<string, string>;
    preferences: string[];
    mentioned_products: string[];
    session_summaries: Array<{ summary: string; time: string }>;
    interaction_count: number;
  };
}

export interface HealthStatus {
  status: string;
  bm25_docs: number;
  active_sessions: number;
}

export interface KnowledgeUploadResponse {
  msg: string;
  filename: string;
}

// ==================== SSE 事件类型（Phase 2 扩展） ====================

export type SSEEvent =
  | { type: 'token'; content: string }
  // 工具审批（Phase 1）
  | { type: 'approval_required'; approval_id: string; tool: string; args: Record<string, any>; server: string; thread_id: string; hierarchy: 'tool' }
  // 计划审批（Phase 2）
  | { type: 'plan_review'; hierarchy: 'plan'; plan: PlanDict }
  // 审查升级（Phase 2）
  | { type: 'review_escalation'; hierarchy: 'review'; step_id: string; description: string; last_result: string; review_issues: string; retries_exhausted: number }
  // 计划进度事件（Phase 2）
  | { type: 'plan_created'; plan_id: string; summary: string; task_count: number }
  | { type: 'task_started'; task_id: string; description: string }
  | { type: 'task_completed'; task_id: string; result_preview: string }
  | { type: 'task_failed'; task_id: string; error: string }
  | { type: 'plan_completed'; summary: string }
  // 思考过程事件（ReAct 推理 / 工具调用 / 观察）
  | { type: 'thinking'; text: string }
  | { type: 'tool_call'; tool: string; args: Record<string, any> }
  | { type: 'tool_result'; tool: string; result_preview: string }
  | { type: 'done'; meta: Record<string, any> };

// ==================== Plan-and-Execute 类型（Phase 2 新增） ====================

export interface PlanDict {
  id: string;
  goal: string;
  summary: string;
  tasks: TaskDict[];
  execution_order: string[];
  status: string;
  progress: number;
}

export interface TaskDict {
  id: string;
  description: string;
  type: string;
  dependencies: string[];
  dependents: string[];
  status: 'PENDING' | 'RUNNING' | 'COMPLETED' | 'FAILED' | 'SKIPPED';
  result: string;
  error: string;
}

// 审批层级枚举（前端根据层级渲染不同 UI）
export type ApprovalHierarchy = 'tool' | 'plan' | 'review';

export interface ApprovalRequest {
  approval_id: string;
  tool: string;
  args: Record<string, any>;
  server: string;
  thread_id: string;
  status: 'pending' | 'approved' | 'rejected' | 'expired';
  // review_escalation 扩展字段
  hierarchy?: ApprovalHierarchy;
  step_id?: string;
  description?: string;
  last_result?: string;
  review_issues?: string;
  retries_exhausted?: number;
}

export interface ApprovalResult {
  user_id: string;
  decision: 'approved' | 'rejected' | 'approve_all';
  reject_reason?: string;
}

// ==================== Plan/Multi-Agent 进度类型（Phase 2 前端渲染） ====================

export type TaskProgressStatus = 'PENDING' | 'RUNNING' | 'COMPLETED' | 'FAILED';

export interface TaskProgress {
  id: string;
  description: string;
  type: string;
  status: TaskProgressStatus;
  resultPreview?: string;
  error?: string;
}

export interface PlanProgress {
  planId: string;
  summary: string;
  taskCount: number;
  tasks: TaskProgress[];
  status: 'planning' | 'executing' | 'completed';
  finalSummary?: string;
}

// ==================== 前端内部类型 ====================

// 思考过程步骤（ReAct）
export interface ThinkingStep {
  kind: 'thinking' | 'tool';
  text?: string;               // kind='thinking' 的推理文本
  tool?: string;               // kind='tool' 的工具名
  args?: Record<string, any>;  // kind='tool' 的调用参数
  resultPreview?: string;      // kind='tool' 的观察结果（由 tool_result 事件回填）
}

// 来源引用
export interface SourceCitation {
  document: string;    // 文档名
  page?: number | null;
  section?: string;
}

export interface UIMessage {
  id: string;
  role: 'user' | 'assistant';
  content: string;
  timestamp: string;
  imageCount?: number;
  mode?: string;
  verification?: VerificationInfo;
  turnCount?: number;
  imageDesc?: string;
  detectedProducts?: string[];
  planProgress?: PlanProgress;  // Plan/Multi-Agent 执行进度
  thinkingSteps?: ThinkingStep[];  // ReAct 思考过程
  sources?: SourceCitation[];  // 来源引用
}
