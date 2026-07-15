import { useState } from 'react';
import { Activity, FileText, Cpu, Image, Smartphone, ChevronDown, CheckCircle, AlertTriangle } from 'lucide-react';
import type { HealthStatus, VerificationInfo } from '../types';

interface Props {
  health: HealthStatus | null;
  healthError: string;
  lastVerification: VerificationInfo | null;
  lastTurnCount: number | null;
  sessionId: string | null;
  lastImageDesc: string | null;
  lastDetectedProducts: string[] | null;
}

function Empty({ text }: { text: string }) {
  return <span className="text-fg-subtle text-[11px]">{text}</span>;
}

function SectionTitle({ icon: Icon, label }: { icon: React.ComponentType<{ size?: number }>; label: string }) {
  return (
    <div className="flex items-center gap-1.5 text-[10px] uppercase tracking-widest text-fg-subtle mb-2 font-medium">
      <Icon size={10} />
      {label}
    </div>
  );
}

export default function DiagnosticsPanel({
  health, healthError, lastVerification,
  lastTurnCount, sessionId, lastImageDesc, lastDetectedProducts,
}: Props) {
  const [expanded, setExpanded] = useState(false);
  const healthy = health?.status === 'healthy';
  const passed = lastVerification ? lastVerification.pass !== false : null;

  return (
    <div style={{ animationDelay: '0.14s' }} className="shrink-0 glass border border-border rounded-2xl shadow-card overflow-hidden animate-rise">
      {/* 概要 banner —— 默认一行 */}
      <button
        onClick={() => setExpanded((v) => !v)}
        className="w-full flex items-center gap-3 px-4 py-2.5 text-left hover:bg-surface-2/50 transition-colors"
        aria-expanded={expanded}
        title={expanded ? '收起诊断详情' : '展开诊断详情'}
      >
        <div className="flex items-center gap-1.5 shrink-0">
          <Activity size={14} className="text-primary" />
          <span className="text-xs font-semibold text-fg">诊断</span>
        </div>

        {/* 概要 chips */}
        <div className="flex items-center gap-1.5 flex-1 min-w-0 overflow-hidden text-[11px]">
          {/* 系统状态 */}
          <span className="inline-flex items-center gap-1 px-2 py-0.5 rounded-full bg-surface-2 border border-border shrink-0">
            <span className={`w-1.5 h-1.5 rounded-full ${healthError || !healthy ? 'bg-danger' : 'bg-success animate-pulse-dot'}`} />
            <span className="text-fg-muted">
              {healthError ? '异常' : health ? health.status : '加载中'}
            </span>
          </span>

          {health && (
            <span className="hidden sm:inline text-fg-subtle shrink-0">
              {health.bm25_docs} 知识 · {health.active_sessions} 会话
            </span>
          )}

          {lastTurnCount != null && (
            <span className="inline-flex items-center gap-1 px-2 py-0.5 rounded-full bg-surface-2 border border-border shrink-0 text-fg-muted">
              第 {lastTurnCount} 轮
            </span>
          )}

          {passed != null && (
            <span className={`inline-flex items-center gap-1 px-2 py-0.5 rounded-full border shrink-0 ${
              passed ? 'bg-success/10 text-success border-success/20' : 'bg-danger/10 text-danger border-danger/20'
            }`}>
              {passed ? <CheckCircle size={9} /> : <AlertTriangle size={9} />}
              验证{lastVerification?.score != null ? ` ${lastVerification.score}分` : passed ? '通过' : '未过'}
            </span>
          )}

          {sessionId && (
            <span className="font-mono text-fg-subtle truncate min-w-0 hidden md:inline">{sessionId}</span>
          )}
        </div>

        <ChevronDown
          size={15}
          className={`shrink-0 text-fg-subtle transition-transform ${expanded ? 'rotate-180' : ''}`}
        />
      </button>

      {/* 展开详情 */}
      {expanded && (
        <div className="border-t border-border px-4 py-3.5 grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-x-5 gap-y-4 animate-rise">
          {/* 验证 */}
          <section className="min-w-0">
            <SectionTitle icon={FileText} label="验证结果" />
            {lastVerification ? (
              <div className="bg-surface-2 rounded-md p-2.5">
                <pre className="text-[10px] font-mono text-fg-muted whitespace-pre-wrap leading-relaxed max-h-40 overflow-y-auto">
                  {JSON.stringify(lastVerification, null, 1)}
                </pre>
              </div>
            ) : <Empty text="暂无验证数据" />}
          </section>

          {/* 会话信息 */}
          <section className="min-w-0">
            <SectionTitle icon={Activity} label="会话信息" />
            <div className="space-y-1 text-[11px]">
              {lastTurnCount != null ? (
                <div className="flex justify-between py-0.5">
                  <span className="text-fg-muted">当前轮次</span>
                  <span className="font-mono text-primary font-medium">{lastTurnCount}</span>
                </div>
              ) : null}
              {sessionId ? (
                <div className="flex justify-between gap-2 py-0.5">
                  <span className="text-fg-muted shrink-0">会话 ID</span>
                  <span className="font-mono text-fg-muted text-[10px] truncate">{sessionId}</span>
                </div>
              ) : null}
              {lastTurnCount == null && !sessionId && <Empty text="暂无会话数据" />}
            </div>
          </section>

          {/* 图片识别 */}
          <section className="min-w-0">
            <SectionTitle icon={Image} label="图片识别" />
            {lastImageDesc ? (
              <p className="text-[11px] text-fg-muted bg-surface-2 rounded p-2 leading-relaxed max-h-28 overflow-y-auto">{lastImageDesc}</p>
            ) : <Empty text="暂无图片数据" />}
            {lastDetectedProducts && lastDetectedProducts.length > 0 && (
              <div className="mt-2">
                <SectionTitle icon={Smartphone} label="检测产品" />
                <div className="flex flex-wrap gap-1">
                  {lastDetectedProducts.map((p, i) => (
                    <span key={i} className="px-1.5 py-0.5 bg-accent/10 text-accent rounded text-[10px] border border-accent/20">
                      {p}
                    </span>
                  ))}
                </div>
              </div>
            )}
          </section>

          {/* 系统状态 */}
          <section className="min-w-0">
            <SectionTitle icon={Cpu} label="系统状态" />
            {healthError ? (
              <div className="text-[11px] text-danger bg-danger/10 rounded p-2">{healthError}</div>
            ) : health ? (
              <div className="space-y-1 text-[11px]">
                <div className="flex justify-between py-0.5">
                  <span className="text-fg-muted">状态</span>
                  <span className={`font-medium ${health.status === 'healthy' ? 'text-success' : 'text-danger'}`}>
                    {health.status}
                  </span>
                </div>
                <div className="flex justify-between py-0.5">
                  <span className="text-fg-muted">BM25 文档</span>
                  <span className="font-mono text-fg">{health.bm25_docs}</span>
                </div>
                <div className="flex justify-between py-0.5">
                  <span className="text-fg-muted">活跃会话</span>
                  <span className="font-mono text-fg">{health.active_sessions}</span>
                </div>
              </div>
            ) : (
              <Empty text="加载中..." />
            )}
          </section>
        </div>
      )}
    </div>
  );
}
