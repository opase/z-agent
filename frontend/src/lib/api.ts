import type {
  ChatRequest,
  ChatResponse,
  SessionSummary,
  SessionHistory,
  UserProfile,
  HealthStatus,
  KnowledgeUploadResponse,
  ApprovalResult,
  SSEEvent,
} from '../types';

const BASE = import.meta.env.VITE_API_BASE_URL || 'http://127.0.0.1:8080';

async function request<T>(path: string, init?: RequestInit): Promise<T> {
  const url = `${BASE}${path}`;
  const res = await fetch(url, init);
  if (!res.ok) {
    const body = await res.text();
    throw new Error(`[${res.status}] ${body || res.statusText}`);
  }
  return res.json();
}

// ==================== 审批 API ====================

/** 发送审批决定 — 批准或拒绝 MCP 工具调用 */
export async function resumeApproval(threadId: string, result: ApprovalResult): Promise<any> {
  return request(`/approval/${encodeURIComponent(threadId)}/resume`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(result),
  });
}

// ==================== 流式请求底层 ====================

export interface StreamResult {
  text: string;
  meta: Record<string, any> | null;
}

function parseStreamMeta(raw: string): StreamResult {
  const marker = /__CA_META__(\{.*?\})__CA_META_END__/s;
  const match = raw.match(marker);
  if (match) {
    try {
      const meta = JSON.parse(match[1]);
      const text = raw.replace(marker, '').trim();
      return { text, meta };
    } catch {
      // JSON 解析失败，放弃元数据
    }
  }
  return { text: raw.trim(), meta: null };
}

async function streamFetch(
  url: string,
  init: RequestInit,
  onToken: (token: string) => void,
  signal?: AbortSignal,
  onEvent?: (event: SSEEvent) => void,
): Promise<StreamResult> {
  const res = await fetch(url, { ...init, signal });
  if (!res.ok) {
    throw new Error(`[${res.status}] ${await res.text()}`);
  }
  const reader = res.body!.getReader();
  const decoder = new TextDecoder();
  let raw = '';
  let accumulatedText = '';
  let doneMeta: Record<string, any> | null = null;
  let sseBuffer = '';
  let isSSE = false;  // 首次检测到 "data: " 后标记为新格式

  while (true) {
    const { done, value } = await reader.read();
    if (done) break;
    const chunk = decoder.decode(value, { stream: true });
    raw += chunk;

    // 首次 chunk 检测是否为 SSE 格式
    if (!isSSE && raw.trimStart().startsWith('data: ')) {
      isSSE = true;
    }

    if (isSSE) {
      sseBuffer += chunk;
      const lines = sseBuffer.split('\n');
      sseBuffer = lines.pop() || '';  // 保留不完整行
      for (const line of lines) {
        if (line.startsWith('data: ')) {
          try {
            const event: SSEEvent = JSON.parse(line.slice(6));
            if (event.type === 'token') {
              accumulatedText += event.content;
              onToken(event.content);
            } else if (event.type === 'done') {
              doneMeta = event.meta;
            } else {
              // 其他事件（approval_required, plan_created, task_started 等）→ 回调
              if (onEvent) onEvent(event);
            }
          } catch {
            // JSON 解析失败，作为纯文本 token 输出
            if (line.slice(6).trim()) {
              onToken(line.slice(6));
              accumulatedText += line.slice(6);
            }
          }
        }
      }
    } else {
      // 旧格式：纯文本 token + __CA_META__ 标记
      const MARKER = '__CA_META__';
      const idx = raw.indexOf(MARKER);
      if (idx >= 0) {
        // 只发送标记之前的内容
        const before = raw.substring(0, idx);
        const newText = before.slice(accumulatedText.length);
        if (newText) {
          onToken(newText);
          accumulatedText = before;
        }
        // 不再发送后续内容
      } else {
        // 增量发送新 token
        const newText = raw.slice(accumulatedText.length);
        if (newText) {
          onToken(newText);
          accumulatedText += newText;
        }
      }
    }
  }

  // 处理 SSE 缓冲中剩余的 done 事件
  if (isSSE && sseBuffer) {
    for (const line of sseBuffer.split('\n')) {
      if (line.startsWith('data: ')) {
        try {
          const event: SSEEvent = JSON.parse(line.slice(6));
          if (event.type === 'done') {
            doneMeta = event.meta;
          } else if (onEvent) {
            onEvent(event);
          }
        } catch { /* skip */ }
      }
    }
  }

  // 新格式返回累积文本 + done meta，旧格式回退 parseStreamMeta
  if (isSSE) {
    return { text: accumulatedText, meta: doneMeta };
  }
  return parseStreamMeta(raw);
}

// ==================== 对话 ====================

export async function sendChat(req: ChatRequest): Promise<ChatResponse> {
  return request<ChatResponse>('/chat', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(req),
  });
}

export async function sendChatImage(form: FormData): Promise<ChatResponse> {
  return request<ChatResponse>('/chat/image', {
    method: 'POST',
    body: form,
  });
}

/** 流式文本对话——逐 token 回调 + 事件回调，返回文本+元数据 */
export async function streamChat(
  req: ChatRequest,
  onToken: (token: string) => void,
  signal?: AbortSignal,
  onEvent?: (event: SSEEvent) => void,
): Promise<StreamResult> {
  return streamFetch(
    `${BASE}/chat/stream`,
    {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(req),
    },
    onToken,
    signal,
    onEvent,
  );
}

/** 流式图片对话——逐 token 回调 + 事件回调，返回文本+元数据 */
export async function streamChatImage(
  form: FormData,
  onToken: (token: string) => void,
  signal?: AbortSignal,
  onEvent?: (event: SSEEvent) => void,
): Promise<StreamResult> {
  return streamFetch(
    `${BASE}/chat/image/stream`,
    { method: 'POST', body: form },
    onToken,
    signal,
    onEvent,
  );
}

export async function endSession(sessionId: string, userId: string): Promise<{ msg: string }> {
  return request(`/chat/${sessionId}/end?user_id=${encodeURIComponent(userId)}`, {
    method: 'POST',
  });
}

export async function deleteSession(sessionId: string): Promise<{ msg: string }> {
  return request(`/chat/${encodeURIComponent(sessionId)}`, {
    method: 'DELETE',
  });
}

export async function getSessionHistory(sessionId: string): Promise<SessionHistory> {
  return request(`/chat/${sessionId}/history`);
}

// ==================== 知识库 ====================

export async function uploadKnowledge(file: File): Promise<KnowledgeUploadResponse> {
  const form = new FormData();
  form.append('file', file);
  return request<KnowledgeUploadResponse>('/knowledge/upload', {
    method: 'POST',
    body: form,
  });
}

// ==================== 用户 & 会话 ====================

export async function getUserProfile(userId: string): Promise<UserProfile> {
  return request(`/users/${encodeURIComponent(userId)}/profile`);
}

export async function listSessions(): Promise<{ sessions: SessionSummary[] }> {
  return request('/users/sessions');
}

// ==================== 健康检查 ====================

export async function getHealth(): Promise<HealthStatus> {
  return request('/health');
}
