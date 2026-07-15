import { useState, useEffect, useCallback } from 'react';
import Toolbar from './components/Toolbar';
import SessionSidebar from './components/SessionSidebar';
import ChatPanel from './components/ChatPanel';
import DiagnosticsPanel from './components/DiagnosticsPanel';
import KnowledgePanel from './components/KnowledgePanel';
import ApprovalModal from './components/ApprovalModal';
import {
  streamChat, streamChatImage, endSession, deleteSession, getSessionHistory,
  getUserProfile, listSessions, getHealth,
  resumeApproval,
} from './lib/api';
import type {
  UIMessage, SessionSummary, UserProfile, HealthStatus,
  ApprovalRequest, SSEEvent,
  PlanProgress, TaskProgress,
} from './types';

function now() {
  return new Date().toLocaleTimeString('zh-CN', { hour: '2-digit', minute: '2-digit' });
}

export default function App() {
  // ---- 状态 ----
  const [userId, setUserId] = useState('default');
  const [sessionId, setSessionId] = useState<string | null>(null);
  const [messages, setMessages] = useState<UIMessage[]>([]);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState('');

  // 左侧面板
  const [sessions, setSessions] = useState<SessionSummary[]>([]);
  const [sessionsLoading, setSessionsLoading] = useState(false);
  const [profile, setProfile] = useState<UserProfile | null>(null);
  const [profileLoading, setProfileLoading] = useState(false);

  // 诊断（流式模式下只保留 session 和轮次信息）
  const [health, setHealth] = useState<HealthStatus | null>(null);
  const [healthError, setHealthError] = useState('');
  const [lastResult, setLastResult] = useState<Record<string, any> | null>(null);

  // 审批队列（HITL — Phase 1 新增）
  const [approvalQueue, setApprovalQueue] = useState<ApprovalRequest[]>([]);
  const [currentApproval, setCurrentApproval] = useState<ApprovalRequest | null>(null);

  // 执行模式（Phase 2 前端选择器）
  const [execMode, setExecMode] = useState<string>('auto');

  // Plan/Multi-Agent 进度映射（Phase 2 前端渲染）
  const [planProgressMap, setPlanProgressMap] = useState<Record<string, PlanProgress>>({});

  // ---- 数据刷新 ----
  const refreshHealth = useCallback(async () => {
    try {
      setHealthError('');
      const h = await getHealth();
      setHealth(h);
    } catch (e: any) {
      setHealthError(e.message || '无法获取健康状态');
    }
  }, []);

  const refreshSessions = useCallback(async () => {
    setSessionsLoading(true);
    try {
      const data = await listSessions();
      setSessions(data.sessions);
    } catch {
      // 静默失败
    } finally {
      setSessionsLoading(false);
    }
  }, []);

  const refreshProfile = useCallback(async (uid: string) => {
    setProfileLoading(true);
    try {
      const p = await getUserProfile(uid);
      setProfile(p);
    } catch {
      setProfile(null);
    } finally {
      setProfileLoading(false);
    }
  }, []);

  // 初始化
  useEffect(() => {
    refreshHealth();
    refreshSessions();
    refreshProfile(userId);
  }, []);

  const handleUserIdChange = (uid: string) => {
    setUserId(uid);
    refreshProfile(uid);
  };

  useEffect(() => {
    const timer = setInterval(refreshHealth, 30000);
    return () => clearInterval(timer);
  }, [refreshHealth]);

  // ---- 消息处理（流式）----
  const appendUserMessage = (text: string, imageCount = 0): UIMessage => ({
    id: `u-${Date.now()}`,
    role: 'user',
    content: text,
    timestamp: now(),
    imageCount,
  });

  // Plan/Multi-Agent 进度事件处理
  const handlePlanProgress = useCallback((event: SSEEvent, assistantId: string) => {
    const updateProgress = (fn: (prev: PlanProgress | undefined) => PlanProgress) => {
      setPlanProgressMap((prev) => ({
        ...prev,
        [assistantId]: fn(prev[assistantId]),
      }));
    };

    if (event.type === 'plan_created') {
      updateProgress(() => ({
        planId: event.plan_id,
        summary: event.summary,
        taskCount: event.task_count,
        tasks: [],
        status: 'executing',
      }));
      // 同时给消息打上进度引用
      setMessages((prev) =>
        prev.map((m) => m.id === assistantId
          ? { ...m, content: `📋 计划已创建：${event.summary}（${event.task_count} 个任务）` }
          : m),
      );
    } else if (event.type === 'task_started') {
      updateProgress((prev) => {
        if (!prev) return prev!;
        return {
          ...prev,
          tasks: [...prev.tasks.filter(t => t.id !== event.task_id), {
            id: event.task_id,
            description: event.description,
            type: '',
            status: 'RUNNING',
          }],
        };
      });
    } else if (event.type === 'task_completed') {
      updateProgress((prev) => {
        if (!prev) return prev!;
        return {
          ...prev,
          tasks: prev.tasks.map(t => t.id === event.task_id
            ? { ...t, status: 'COMPLETED', resultPreview: event.result_preview }
            : t),
        };
      });
    } else if (event.type === 'task_failed') {
      updateProgress((prev) => {
        if (!prev) return prev!;
        return {
          ...prev,
          tasks: prev.tasks.map(t => t.id === event.task_id
            ? { ...t, status: 'FAILED', error: event.error }
            : t),
        };
      });
    } else if (event.type === 'plan_completed') {
      updateProgress((prev) => prev ? { ...prev, status: 'completed', finalSummary: event.summary } : prev!);
    }
  }, []);

  // 思考过程事件处理（ReAct 推理 / 工具调用 / 观察）
  const handleThinking = useCallback((event: SSEEvent, assistantId: string) => {
    setMessages((prev) =>
      prev.map((m) => {
        if (m.id !== assistantId) return m;
        const steps = m.thinkingSteps ? [...m.thinkingSteps] : [];
        if (event.type === 'thinking') {
          steps.push({ kind: 'thinking', text: event.text });
        } else if (event.type === 'tool_call') {
          steps.push({ kind: 'tool', tool: event.tool, args: event.args });
        } else if (event.type === 'tool_result') {
          // 回填到最近一个同名、尚无结果的工具步骤
          for (let i = steps.length - 1; i >= 0; i--) {
            if (steps[i].kind === 'tool' && steps[i].tool === event.tool && steps[i].resultPreview === undefined) {
              steps[i] = { ...steps[i], resultPreview: event.result_preview };
              break;
            }
          }
        }
        return { ...m, thinkingSteps: steps };
      }),
    );
  }, []);

  // 统一 SSE 事件处理（审批 + 审查升级 + 思考过程 + 计划进度）
  const handleSSEEvent = useCallback((event: SSEEvent, assistantId: string) => {
    if (event.type === 'approval_required') {
      const approvalReq: ApprovalRequest = {
        approval_id: event.approval_id,
        tool: event.tool,
        args: event.args,
        server: event.server,
        thread_id: event.thread_id,
        status: 'pending',
        hierarchy: 'tool',
      };
      setApprovalQueue((prev) => {
        if (prev.length === 0 && !currentApproval) {
          setCurrentApproval(approvalReq);
          return prev;
        }
        return [...prev, approvalReq];
      });
    } else if (event.type === 'review_escalation') {
      const escalationReq: ApprovalRequest = {
        approval_id: `review-${event.step_id}`,
        tool: event.step_id,
        args: {},
        server: 'Multi-Agent Reviewer',
        thread_id: sessionId || 'default',
        status: 'pending',
        hierarchy: 'review',
        step_id: event.step_id,
        description: event.description,
        last_result: event.last_result,
        review_issues: event.review_issues,
        retries_exhausted: event.retries_exhausted,
      };
      setApprovalQueue((prev) => {
        if (prev.length === 0 && !currentApproval) {
          setCurrentApproval(escalationReq);
          return prev;
        }
        return [...prev, escalationReq];
      });
    } else if (event.type === 'plan_review') {
      const planReq: ApprovalRequest = {
        approval_id: `plan-${event.plan?.id || 'unknown'}`,
        tool: '执行计划审批',
        args: event.plan || {},
        server: 'Planner',
        thread_id: sessionId || 'default',
        status: 'pending',
        hierarchy: 'plan',
        description: event.plan?.summary || event.plan?.goal || '',
      };
      setApprovalQueue((prev) => {
        if (prev.length === 0 && !currentApproval) {
          setCurrentApproval(planReq);
          return prev;
        }
        return [...prev, planReq];
      });
    } else if (event.type === 'thinking' || event.type === 'tool_call' || event.type === 'tool_result') {
      handleThinking(event, assistantId);
    } else {
      // plan_created / task_started / task_completed / task_failed / plan_completed
      handlePlanProgress(event, assistantId);
    }
  }, [currentApproval, handlePlanProgress, handleThinking, sessionId]);

  // 纯文本发送（流式）
  const handleSendText = async (question: string) => {
    setError('');
    setLoading(true);

    const userMsg = appendUserMessage(question);
    const assistantId = `a-${Date.now()}`;
    const assistantPlaceholder: UIMessage = {
      id: assistantId, role: 'assistant', content: '', timestamp: now(),
    };
    setMessages((prev) => [...prev, userMsg, assistantPlaceholder]);

    let answer = '';
    try {
      const result = await streamChat(
        { question, session_id: sessionId, user_id: userId, mode: execMode },
        (token) => {
          answer += token;
          setMessages((prev) =>
            prev.map((m) => (m.id === assistantId ? { ...m, content: answer } : m)),
          );
        },
        undefined,  // signal
        (event: SSEEvent) => handleSSEEvent(event, assistantId),
      );

      // 从流末尾提取元数据
      const meta = result.meta || {};
      const sid = meta.session_id || sessionId;
      if (sid && sid !== sessionId) setSessionId(sid);
      const pp = planProgressMap[assistantId];
      setMessages((prev) =>
        prev.map((m) =>
          m.id === assistantId
            ? { ...m, content: result.text, mode: meta.mode, turnCount: meta.turn_count, verification: meta.verification, planProgress: pp, sources: meta.sources || [] }
            : m,
        ),
      );
      setLastResult({
        answer: result.text,
        mode: meta.mode,
        verification: meta.verification,
        session_id: sid,
        turn_count: meta.turn_count,
        image_desc: meta.image_desc,
        detected_products: meta.detected_products,
      });
    } catch (e: any) {
      setError(e.message || '请求失败');
      // 移除空占位
      setMessages((prev) => prev.filter((m) => m.id !== assistantId));
      setPlanProgressMap((prev) => { const n = { ...prev }; delete n[assistantId]; return n; });
    } finally {
      setLoading(false);
      refreshSessions();
    }
  };

  // 图片发送（流式）
  const handleSendImage = async (question: string, files: File[]) => {
    setError('');
    setLoading(true);

    const userMsg = appendUserMessage(question, files.length);
    const assistantId = `a-${Date.now()}`;
    const assistantPlaceholder: UIMessage = {
      id: assistantId, role: 'assistant', content: '', timestamp: now(),
    };
    setMessages((prev) => [...prev, userMsg, assistantPlaceholder]);

    const form = new FormData();
    form.append('question', question);
    form.append('user_id', userId);
    if (sessionId) form.append('session_id', sessionId);
    files.forEach((f) => form.append('images', f));

    let answer = '';
    try {
      const result = await streamChatImage(form,
        (token) => {
          answer += token;
          setMessages((prev) =>
            prev.map((m) => (m.id === assistantId ? { ...m, content: answer } : m)),
          );
        },
        undefined,  // signal
        (event: SSEEvent) => handleSSEEvent(event, assistantId),
      );

      const meta = result.meta || {};
      const sid = meta.session_id || sessionId;
      if (sid && sid !== sessionId) setSessionId(sid);
      setMessages((prev) =>
        prev.map((m) =>
          m.id === assistantId
            ? { ...m, content: result.text, mode: meta.mode, turnCount: meta.turn_count, verification: meta.verification, sources: meta.sources || [] }
            : m,
        ),
      );
      setLastResult({
        answer: result.text,
        mode: meta.mode,
        verification: meta.verification,
        session_id: sid,
        turn_count: meta.turn_count,
        image_desc: meta.image_desc,
        detected_products: meta.detected_products,
        image_count: files.length,
      });
    } catch (e: any) {
      setError(e.message || '请求失败');
      setMessages((prev) => prev.filter((m) => m.id !== assistantId));
    } finally {
      setLoading(false);
      refreshSessions();
    }
  };

  // 结束会话
  const handleEndSession = async () => {
    if (!sessionId) return;
    try {
      await endSession(sessionId, userId);
    } catch {
      // 即使失败也清空本地
    }
    setSessionId(null);
    setLastResult(null);
    refreshSessions();
    refreshProfile(userId);
  };

  // 新建会话
  const handleNewSession = () => {
    setSessionId(null);
    setMessages([]);
    setLastResult(null);
    setError('');
  };

  // 删除会话
  const handleDeleteSession = async (sid: string) => {
    try {
      await deleteSession(sid);
    } catch {
      // 静默
    }
    if (sessionId === sid) {
      setSessionId(null);
      setMessages([]);
      setLastResult(null);
    }
    refreshSessions();
  };

  // 清空界面
  const handleClearMessages = () => {
    setMessages([]);
    setLastResult(null);
    setError('');
  };

  // 选择历史会话
  const handleSelectSession = async (sid: string) => {
    try {
      const history = await getSessionHistory(sid);
      const uiMessages: UIMessage[] = history.messages.map((m, i) => ({
        id: `h-${sid}-${i}`,
        role: (m.role === 'user' ? 'user' : 'assistant') as 'user' | 'assistant',
        content: m.content,
        timestamp: '',
        imageCount: m.image_count || 0,
      }));
      setMessages(uiMessages);
      setSessionId(sid);
      setLastResult(null);
      setError('');
    } catch {
      // 静默失败
    }
  };

  const resolveApproval = async (decision: 'approved' | 'rejected' | 'approve_all', reason?: string) => {
    if (!currentApproval) return;
    try {
      await resumeApproval(currentApproval.thread_id, {
        user_id: userId,
        decision,
        reject_reason: reason,
      });
    } catch (e: any) {
      console.error('审批失败:', e);
    }
    setApprovalQueue((prev) => {
      const next = prev.slice(1);
      setCurrentApproval(next.length > 0 ? next[0] : null);
      return next;
    });
  };

  const handleApprove = () => resolveApproval('approved');
  const handleApproveAll = () => resolveApproval('approve_all');
  const handleReject = (reason?: string) => resolveApproval('rejected', reason);

  const handleApprovalClose = () => {
    // 关闭弹窗（不做出决定）—— 审批保留在队列中，可通过通知横幅重新打开
    setCurrentApproval(null);
  };

  const handleKnowledgeUploaded = () => {
    refreshHealth();
  };

  return (
    <div className="relative h-full flex flex-col">
      {/* 渐变网格背景层 */}
      <div className="app-backdrop" aria-hidden="true" />

      <Toolbar
        health={health}
        healthError={healthError}
        userId={userId}
        onRefresh={refreshHealth}
      />

      <div className="relative z-10 flex-1 flex min-h-0 gap-3 px-3 pt-3 pb-3">
        <SessionSidebar
          userId={userId}
          onUserIdChange={handleUserIdChange}
          sessions={sessions}
          sessionsLoading={sessionsLoading}
          onRefreshSessions={refreshSessions}
          currentSessionId={sessionId}
          onNewSession={handleNewSession}
          onDeleteSession={handleDeleteSession}
          onSelectSession={handleSelectSession}
          profile={profile}
          profileLoading={profileLoading}
        />

        {/* 中间栏：诊断概要条（顶）+ 聊天（填充） */}
        <div className="flex-1 flex flex-col min-w-0 min-h-0 gap-3">
          <DiagnosticsPanel
            health={health}
            healthError={healthError}
            lastVerification={lastResult?.verification ?? null}
            lastTurnCount={lastResult?.turn_count ?? null}
            sessionId={sessionId}
            lastImageDesc={lastResult?.image_desc ?? null}
            lastDetectedProducts={lastResult?.detected_products ?? null}
          />

          <ChatPanel
            messages={messages}
            sessionId={sessionId}
            userId={userId}
            loading={loading}
            error={error}
            execMode={execMode}
            onModeChange={setExecMode}
            onSendText={handleSendText}
            onSendImage={handleSendImage}
            onEndSession={handleEndSession}
            onClearMessages={handleClearMessages}
          />
        </div>

        {/* 右侧：知识库可折叠侧栏 */}
        <KnowledgePanel onUploaded={handleKnowledgeUploaded} />
      </div>

      {/* 审批弹窗（HITL — Phase 1 新增） */}
      <ApprovalModal
        request={currentApproval}
        isOpen={currentApproval !== null}
        onApprove={handleApprove}
        onApproveAll={handleApproveAll}
        onReject={handleReject}
        onClose={handleApprovalClose}
      />

      {/* 审批队列通知横幅（有待处理审批但弹窗已关闭时显示） */}
      {currentApproval === null && approvalQueue.length > 0 && (
        <div className="fixed bottom-4 right-4 z-40">
          <button
            onClick={() => setCurrentApproval(approvalQueue[0])}
            className="flex items-center gap-2 px-4 py-2.5 bg-brand-gradient text-white rounded-xl shadow-glow lift transition-all text-sm font-medium"
          >
            <svg className="w-4 h-4" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={2}>
              <path strokeLinecap="round" strokeLinejoin="round" d="M15 17h5l-1.405-1.405A2.032 2.032 0 0118 14.158V11a6 6 0 00-5-5.917V4a1 1 0 10-2 0v1.083A6 6 0 006 11v3.159c0 .538-.214 1.055-.595 1.436L4 17h5m6 0v1a3 3 0 11-6 0v-1m6 0H9" />
            </svg>
            待审批 ({approvalQueue.length})
          </button>
        </div>
      )}
    </div>
  );
}
