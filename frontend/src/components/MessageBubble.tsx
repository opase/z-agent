import { useState } from 'react';
import { Bot, User, Image, Tag, CheckCircle, AlertTriangle, ListChecks, Circle, CircleCheck, CircleX, Brain, Wrench, ChevronDown, FileText } from 'lucide-react';
import ReactMarkdown from 'react-markdown';
import remarkGfm from 'remark-gfm';
import type { UIMessage, TaskProgress } from '../types';

const TASK_ICONS: Record<string, string> = {
  PENDING: 'text-gray-300',
  RUNNING: 'text-blue-500',
  COMPLETED: 'text-green-500',
  FAILED: 'text-red-500',
};

const TASK_STATUS_LABEL: Record<string, string> = {
  PENDING: '等待中',
  RUNNING: '执行中',
  COMPLETED: '已完成',
  FAILED: '失败',
};

const MODE_LABELS: Record<string, string> = {
  react: 'ReAct',
  plan: '计划执行',
  multi_agent: '多智能体',
  auto: '自动',
};

interface Props {
  message: UIMessage;
  streaming?: boolean;
}

export default function MessageBubble({ message, streaming }: Props) {
  const isUser = message.role === 'user';
  const [thinkingOpen, setThinkingOpen] = useState(false);

  return (
    <div className={`flex gap-3 animate-fade-in-up ${isUser ? 'justify-end' : ''}`}>
      {!isUser && (
        <div className="w-8 h-8 rounded-xl bg-gradient-to-br from-primary to-accent flex items-center justify-center shrink-0 mt-0.5 shadow-glow">
          <Bot size={15} className="text-white" />
        </div>
      )}

      <div className={`max-w-[80%] ${isUser ? 'order-first' : ''}`}>
        {/* 用户消息 */}
        {isUser ? (
          <div className="bg-brand-gradient text-white text-sm px-3.5 py-2.5 rounded-2xl rounded-tr-md inline-block shadow-glow">
            {message.imageCount ? (
              <div className="flex items-center gap-1.5 mb-1 text-white/70 text-xs">
                <Image size={12} />
                <span>附图 {message.imageCount} 张</span>
              </div>
            ) : null}
            {message.content}
          </div>
        ) : (
          /* 助手消息 */
          <div className="space-y-2">
            {/* 思考过程（ReAct，默认折叠） */}
            {message.thinkingSteps && message.thinkingSteps.length > 0 && (
              <div className="overflow-hidden rounded-xl border border-border bg-surface-2/50">
                <button
                  onClick={() => setThinkingOpen((o) => !o)}
                  className="flex w-full items-center gap-1.5 px-3 py-1.5 text-[11px] text-fg-muted transition-colors hover:bg-surface-2"
                  aria-expanded={thinkingOpen}
                >
                  <Brain size={12} className="text-accent" />
                  <span className="font-medium">思考过程</span>
                  <span className="text-fg-subtle">· {message.thinkingSteps.length} 步</span>
                  <ChevronDown size={12} className={`ml-auto shrink-0 text-fg-subtle transition-transform ${thinkingOpen ? 'rotate-180' : ''}`} />
                </button>
                {thinkingOpen && (
                  <div className="space-y-2 border-t border-border px-3 py-2 animate-rise">
                    {message.thinkingSteps.map((step, i) => (
                      <div key={i} className="text-[11px]">
                        {step.kind === 'thinking' ? (
                          <div className="flex gap-1.5">
                            <Circle size={7} className="mt-1 shrink-0 text-accent" fill="currentColor" />
                            <span className="whitespace-pre-wrap text-fg-muted">{step.text}</span>
                          </div>
                        ) : (
                          <div className="space-y-0.5">
                            <div className="flex items-center gap-1.5">
                              <Wrench size={11} className="shrink-0 text-info" />
                              <span className="font-mono text-fg">{step.tool}</span>
                            </div>
                            {step.args && Object.keys(step.args).length > 0 && (
                              <pre className="ml-4 max-h-24 overflow-x-auto rounded bg-surface px-1.5 py-0.5 text-[10px] font-mono text-fg-subtle">
                                {JSON.stringify(step.args, null, 1)}
                              </pre>
                            )}
                            {step.resultPreview && (
                              <div className="ml-4 line-clamp-3 rounded bg-surface px-1.5 py-0.5 text-[10px] text-fg-muted">
                                → {step.resultPreview}
                              </div>
                            )}
                          </div>
                        )}
                      </div>
                    ))}
                  </div>
                )}
              </div>
            )}

            <div className={`bg-surface/80 border border-border px-4 py-3 rounded-2xl rounded-tl-md shadow-card md-body ${streaming ? 'typing-caret' : ''}`}>
              <ReactMarkdown remarkPlugins={[remarkGfm]}>
                {message.content}
              </ReactMarkdown>
            </div>

            {/* Plan/Multi-Agent 执行进度——任务时间线（Phase 2） */}
            {message.planProgress && message.planProgress.tasks.length > 0 && (() => {
              const pp = message.planProgress;
              const done = pp.tasks.filter((t) => t.status === 'COMPLETED').length;
              const isMulti = message.mode === 'multi_agent';
              const rail = isMulti ? 'border-accent/30' : 'border-primary/25';
              return (
                <div className="overflow-hidden rounded-2xl border border-border bg-surface shadow-card">
                  {/* 头部 */}
                  <div className={`flex items-center gap-2 border-b border-border px-3 py-2 ${isMulti ? 'bg-accent/5' : 'bg-primary/5'}`}>
                    <ListChecks size={14} className={isMulti ? 'text-accent' : 'text-primary'} />
                    <span className="text-xs font-medium text-fg">
                      {pp.status === 'completed'
                        ? '计划完成'
                        : isMulti
                          ? '多智能体协作中'
                          : '计划执行中'}
                    </span>
                    <span className="ml-auto font-mono text-[10px] text-fg-subtle">
                      {done}/{pp.taskCount}
                    </span>
                  </div>
                  {/* 迷你进度条 */}
                  <div className="h-1 w-full bg-surface-2">
                    <div
                      className="h-full rounded-r-full bg-brand-gradient transition-all duration-500"
                      style={{ width: `${pp.taskCount ? (done / pp.taskCount) * 100 : 0}%` }}
                    />
                  </div>
                  {/* 时间线：左竖轨 + 节点 */}
                  <div className={`space-y-3 border-l-2 ${rail} ml-4 py-3 pl-4 pr-3`}>
                    {pp.tasks.map((task) => (
                      <div key={task.id} className="relative">
                        <span className="absolute -left-[25px] top-0.5 flex h-3.5 w-3.5 items-center justify-center">
                          {task.status === 'RUNNING' && (
                            <span className="h-3 w-3 rounded-full bg-info animate-pulse-ring" />
                          )}
                          {task.status === 'COMPLETED' && (
                            <CircleCheck size={14} className="text-success" />
                          )}
                          {task.status === 'FAILED' && (
                            <CircleX size={14} className="text-danger" />
                          )}
                          {task.status === 'PENDING' && (
                            <Circle size={14} className="text-fg-subtle" />
                          )}
                        </span>
                        <div className="min-w-0 text-xs">
                          <span className="text-fg">{task.description}</span>
                          <span className="ml-1.5 text-[10px] text-fg-subtle">
                            {TASK_STATUS_LABEL[task.status]}
                          </span>
                          {task.resultPreview && (
                            <p className="mt-1 line-clamp-2 rounded bg-surface-2 px-1.5 py-0.5 text-[10px] text-fg-muted">
                              {task.resultPreview}
                            </p>
                          )}
                          {task.error && (
                            <p className="mt-1 text-[10px] text-danger">{task.error}</p>
                          )}
                        </div>
                      </div>
                    ))}
                  </div>
                  {/* 最终汇总 */}
                  {pp.finalSummary && (
                    <div className="border-t border-border bg-surface-2 px-3 py-2">
                      <p className="line-clamp-4 whitespace-pre-wrap text-[11px] text-fg-muted">
                        {pp.finalSummary}
                      </p>
                    </div>
                  )}
                </div>
              );
            })()}

            {/* 元信息卡片 */}
            <div className="flex flex-wrap gap-1.5 text-[10px]">
              {message.mode && (
                <span className="inline-flex items-center gap-1 px-2 py-0.5 bg-accent/10 text-accent rounded-full border border-accent/20">
                  <Tag size={9} />
                  {MODE_LABELS[message.mode] || message.mode}
                </span>
              )}

              {message.turnCount != null && (
                <span className="inline-flex items-center gap-1 px-2 py-0.5 bg-surface-2 text-fg-muted rounded-full border border-border">
                  第 {message.turnCount} 轮
                </span>
              )}

              {message.verification && (
                <span className={`inline-flex items-center gap-1 px-2 py-0.5 rounded-full border ${
                  message.verification.pass !== false
                    ? 'bg-success/10 text-success border-success/20'
                    : 'bg-danger/10 text-danger border-danger/20'
                }`}>
                  {message.verification.pass !== false ? (
                    <CheckCircle size={9} />
                  ) : (
                    <AlertTriangle size={9} />
                  )}
                  验证{message.verification.score != null ? `  ${message.verification.score}分` : ''}
                </span>
              )}
            </div>

            {/* 来源引用 */}
            {message.sources && message.sources.length > 0 && (
              <div className="text-[11px] text-fg-muted bg-surface-2 px-2.5 py-1.5 rounded-md border border-border">
                <span className="inline-flex items-center gap-1 text-fg-subtle font-medium mb-1">
                  <FileText size={11} />
                  参考来源 ({message.sources.length})
                </span>
                <div className="space-y-0.5">
                  {message.sources.map((src, i) => (
                    <div key={i} className="flex items-center gap-1.5 text-fg-muted">
                      <span className="text-accent font-medium shrink-0">{src.document}</span>
                      {src.page != null && (
                        <span className="text-fg-subtle">· 第{src.page}页</span>
                      )}
                      {src.section && (
                        <span className="text-fg-subtle truncate">· {src.section}</span>
                      )}
                    </div>
                  ))}
                </div>
              </div>
            )}

            {/* 图片识别信息 */}
            {message.imageDesc && (
              <div className="text-[11px] text-fg-muted bg-surface-2 px-2.5 py-1.5 rounded-md border border-border">
                <span className="text-fg-subtle">图片识别：</span>
                {message.imageDesc}
                {message.detectedProducts && message.detectedProducts.length > 0 && (
                  <span className="ml-1.5 text-accent">
                    检测到：{message.detectedProducts.join('、')}
                  </span>
                )}
              </div>
            )}
          </div>
        )}

        {/* 时间戳 */}
        <div className={`text-[10px] text-fg-subtle mt-0.5 ${isUser ? 'text-right' : ''}`}>
          {message.timestamp}
        </div>
      </div>

      {isUser && (
        <div className="w-8 h-8 rounded-xl bg-brand-gradient flex items-center justify-center shrink-0 mt-0.5 shadow-glow">
          <User size={15} className="text-white" />
        </div>
      )}
    </div>
  );
}
