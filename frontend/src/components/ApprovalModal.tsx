import { useState, useEffect } from 'react';
import { X, Check, AlertTriangle, FileText, ShieldAlert } from 'lucide-react';
import type { ApprovalRequest } from '../types';

interface Props {
  request: ApprovalRequest | null;
  isOpen: boolean;
  onApprove: () => void;
  onApproveAll: () => void;
  onReject: (reason?: string) => void;
  onClose: () => void;
}

/** 审批弹窗——根据 hierarchy 渲染不同 UI：工具审批 / 审查升级 / 计划审批 */
export default function ApprovalModal({ request, isOpen, onApprove, onApproveAll, onReject, onClose }: Props) {
  const [rejectReason, setRejectReason] = useState('');
  const [elapsed, setElapsed] = useState(0);

  useEffect(() => {
    if (!isOpen) { setElapsed(0); return; }
    const timer = setInterval(() => setElapsed((p) => p + 1), 1000);
    return () => clearInterval(timer);
  }, [isOpen]);

  if (!isOpen || !request) return null;

  const fmtTime = (s: number) => `${Math.floor(s / 60)}:${(s % 60).toString().padStart(2, '0')}`;
  const isReview = request.hierarchy === 'review';
  const isPlan = request.hierarchy === 'plan';

  // 审查升级模式
  if (isReview) {
    return (
      <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/50 backdrop-blur-sm">
        <div className="bg-surface rounded-xl shadow-pop max-w-lg w-full mx-4 border-t-4 border-warning overflow-hidden">
          {/* 头部 */}
          <div className="px-6 py-4 border-b border-border flex items-center justify-between">
            <div className="flex items-center gap-2">
              <span className="inline-flex items-center justify-center w-7 h-7 rounded-full bg-warning/15">
                <ShieldAlert className="w-4 h-4 text-warning" />
              </span>
              <h3 className="text-lg font-semibold text-fg">审查升级 — 人工裁决</h3>
            </div>
            <button onClick={onClose} className="text-fg-subtle hover:text-fg transition-colors" title="稍后处理">
              <X className="w-5 h-5" />
            </button>
          </div>

          {/* 内容 */}
          <div className="px-6 py-4 space-y-3">
            <div>
              <span className="text-xs font-medium text-fg-muted uppercase tracking-wide">步骤</span>
              <p className="text-sm font-mono text-fg mt-0.5">{request.step_id}</p>
            </div>
            <div>
              <span className="text-xs font-medium text-fg-muted uppercase tracking-wide">任务描述</span>
              <p className="text-sm text-fg mt-0.5">{request.description}</p>
            </div>
            {request.last_result && (
              <div>
                <span className="text-xs font-medium text-fg-muted uppercase tracking-wide">最后执行结果</span>
                <pre className="mt-1 p-2.5 bg-surface-2 rounded-lg text-xs font-mono text-fg-muted max-h-24 overflow-y-auto border border-border whitespace-pre-wrap">
                  {request.last_result.slice(0, 500)}
                </pre>
              </div>
            )}
            <div>
              <span className="text-xs font-medium text-fg-muted uppercase tracking-wide">审查驳回原因</span>
              <pre className="mt-1 p-2.5 bg-warning/5 rounded-lg text-xs text-fg max-h-24 overflow-y-auto border border-warning/20 whitespace-pre-wrap">
                {request.review_issues}
              </pre>
            </div>
            <div className="flex items-center justify-between text-xs text-fg-muted">
              <span>重试次数: {request.retries_exhausted}</span>
              <span>等待: {fmtTime(elapsed)}</span>
            </div>
          </div>

          {/* 底部：接受 / 跳过 */}
          <div className="px-6 py-4 border-t border-border flex gap-2">
            <button
              onClick={() => { onReject(rejectReason || undefined); setRejectReason(''); }}
              className="flex-1 inline-flex items-center justify-center gap-1 px-4 py-2 text-sm font-medium text-fg-muted bg-surface-2 hover:bg-border rounded-lg transition-colors"
            >
              <X size={15} /> 跳过此步骤
            </button>
            <button
              onClick={() => { onApprove(); }}
              className="flex-1 inline-flex items-center justify-center gap-1 px-4 py-2 text-sm font-medium text-white bg-green-700 hover:bg-green-800 rounded-lg transition-colors shadow-sm"
            >
              <Check size={15} /> 接受当前结果
            </button>
          </div>
        </div>
      </div>
    );
  }

  // 计划审批模式
  if (isPlan) {
    return (
      <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/50 backdrop-blur-sm">
        <div className="bg-surface rounded-xl shadow-pop max-w-lg w-full mx-4 border-t-4 border-info overflow-hidden">
          <div className="px-6 py-4 border-b border-border flex items-center justify-between">
            <div className="flex items-center gap-2">
              <span className="inline-flex items-center justify-center w-7 h-7 rounded-full bg-info/15">
                <FileText className="w-4 h-4 text-info" />
              </span>
              <h3 className="text-lg font-semibold text-fg">执行计划审批</h3>
            </div>
            <button onClick={onClose} className="text-fg-subtle hover:text-fg transition-colors" title="稍后处理">
              <X className="w-5 h-5" />
            </button>
          </div>
          <div className="px-6 py-4 space-y-3">
            <p className="text-sm text-fg">{request.description}</p>
            <div className="flex items-center justify-between text-xs text-fg-muted">
              <span>等待: {fmtTime(elapsed)}</span>
            </div>
          </div>
          <div className="px-6 py-4 border-t border-border flex gap-2">
            <button
              onClick={() => { onReject(rejectReason || undefined); setRejectReason(''); }}
              className="flex-1 inline-flex items-center justify-center gap-1 px-4 py-2 text-sm font-medium text-fg-muted bg-surface-2 hover:bg-border rounded-lg transition-colors"
            >
              <X size={15} /> 取消计划
            </button>
            <button
              onClick={() => { onApprove(); }}
              className="flex-1 inline-flex items-center justify-center gap-1 px-4 py-2 text-sm font-medium text-white bg-green-700 hover:bg-green-800 rounded-lg transition-colors shadow-sm"
            >
              <Check size={15} /> 批准计划
            </button>
          </div>
        </div>
      </div>
    );
  }

  // 默认：工具审批模式
  const formatArgs = (args: Record<string, any>) => {
    try { return JSON.stringify(args, null, 2); } catch { return String(args); }
  };

  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/50 backdrop-blur-sm">
      <div className="bg-surface rounded-xl shadow-pop max-w-md w-full mx-4 border-t-4 border-danger overflow-hidden">
        {/* 头部 */}
        <div className="px-6 py-4 border-b border-border flex items-center justify-between">
          <div className="flex items-center gap-2">
            <span className="inline-flex items-center justify-center w-7 h-7 rounded-full bg-danger/15">
              <AlertTriangle className="w-4 h-4 text-danger" />
            </span>
            <h3 className="text-lg font-semibold text-fg">工具调用审批</h3>
          </div>
          <button onClick={onClose} className="text-fg-subtle hover:text-fg transition-colors" title="稍后处理">
            <X className="w-5 h-5" />
          </button>
        </div>

        {/* 内容 */}
        <div className="px-6 py-4 space-y-3">
          <div>
            <span className="text-xs font-medium text-fg-muted uppercase tracking-wide">工具</span>
            <p className="text-sm font-mono text-fg mt-0.5 break-all">{request.tool}</p>
          </div>
          <div>
            <span className="text-xs font-medium text-fg-muted uppercase tracking-wide">MCP 服务器</span>
            <p className="text-sm text-fg mt-0.5">{request.server}</p>
          </div>
          <div>
            <span className="text-xs font-medium text-fg-muted uppercase tracking-wide">调用参数</span>
            <pre className="mt-1 p-2.5 bg-surface-2 rounded-lg text-xs font-mono text-fg-muted max-h-32 overflow-y-auto border border-border">
              {formatArgs(request.args)}
            </pre>
          </div>
          <div className="flex items-center justify-between text-xs text-fg-muted">
            <span>线程: {request.thread_id.slice(0, 8)}...</span>
            <span>等待: {fmtTime(elapsed)}</span>
          </div>

          {/* 拒绝原因 */}
          <div>
            <label className="text-xs font-medium text-fg-muted block mb-1">拒绝原因（可选，将反馈给 LLM）</label>
            <input
              type="text" value={rejectReason}
              onChange={(e) => setRejectReason(e.target.value)}
              placeholder="例：参数风险过高，请换一种方式..."
              className="w-full px-3 py-1.5 text-sm border border-border rounded-lg bg-surface-2 text-fg placeholder:text-fg-subtle focus:ring-2 focus:ring-ring outline-none"
            />
          </div>
        </div>

        {/* 底部三按钮 */}
        <div className="px-6 py-4 border-t border-border flex gap-2">
          <button
            onClick={() => { onReject(rejectReason || undefined); setRejectReason(''); }}
            className="inline-flex items-center gap-1 px-3 py-2 text-sm font-medium text-fg-muted bg-surface-2 hover:bg-border rounded-lg transition-colors"
          >
            <X size={15} /> 拒绝
          </button>
          <button
            onClick={() => { onApprove(); }}
            className="flex-1 inline-flex items-center justify-center gap-1 px-4 py-2 text-sm font-medium text-white bg-green-700 hover:bg-green-800 rounded-lg transition-colors shadow-sm"
          >
            <Check size={15} /> 批准
          </button>
          <button
            onClick={() => { onApproveAll(); }}
            className="flex-1 inline-flex items-center justify-center gap-1 px-4 py-2 text-sm font-medium text-white bg-blue-600 hover:bg-blue-700 rounded-lg transition-colors shadow-sm"
          >
            <Check size={15} /> 批准所有
          </button>
        </div>
      </div>
    </div>
  );
}
