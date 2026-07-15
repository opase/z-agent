import { useState, useRef, useEffect } from 'react';
import {
  Send, ImagePlus, X, Square, Trash2, Loader2, ChevronDown,
} from 'lucide-react';
import type { UIMessage } from '../types';
import MessageBubble from './MessageBubble';
import ZagentLogo from './ZagentLogo';

const MODE_OPTIONS: { value: string; label: string; desc: string }[] = [
  { value: 'auto', label: '自动', desc: '根据任务复杂度自动选择' },
  { value: 'react', label: 'ReAct', desc: '标准思考-行动循环' },
  { value: 'plan', label: 'Plan', desc: '先规划再并行执行' },
  { value: 'multi_agent', label: 'Multi', desc: '多角色协作审查' },
];

interface Props {
  messages: UIMessage[];
  sessionId: string | null;
  userId: string;
  loading: boolean;
  error: string;
  execMode: string;
  onModeChange: (mode: string) => void;
  onSendText: (question: string) => Promise<void>;
  onSendImage: (question: string, files: File[]) => Promise<void>;
  onEndSession: () => void;
  onClearMessages: () => void;
}

export default function ChatPanel({
  messages,
  sessionId,
  userId,
  loading,
  error,
  execMode,
  onModeChange,
  onSendText,
  onSendImage,
  onEndSession,
  onClearMessages,
}: Props) {
  const [input, setInput] = useState('');
  const [imageFiles, setImageFiles] = useState<File[]>([]);
  const [modeMenuOpen, setModeMenuOpen] = useState(false);
  const messagesEnd = useRef<HTMLDivElement>(null);
  const textareaRef = useRef<HTMLTextAreaElement>(null);
  const modeRef = useRef<HTMLDivElement>(null);

  // 点击外部关闭模式菜单
  useEffect(() => {
    const handler = (e: MouseEvent) => {
      if (modeRef.current && !modeRef.current.contains(e.target as Node)) {
        setModeMenuOpen(false);
      }
    };
    if (modeMenuOpen) {
      document.addEventListener('mousedown', handler);
      return () => document.removeEventListener('mousedown', handler);
    }
  }, [modeMenuOpen]);

  useEffect(() => {
    messagesEnd.current?.scrollIntoView({ behavior: 'smooth' });
  }, [messages]);

  const handleSend = async () => {
    const text = input.trim();
    if (!text) return;
    if (loading) return;

    setInput('');
    const files = [...imageFiles];
    setImageFiles([]);

    if (files.length > 0) {
      await onSendImage(text, files);
    } else {
      await onSendText(text);
    }

    textareaRef.current?.focus();
  };

  const handleKeyDown = (e: React.KeyboardEvent) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      handleSend();
    }
  };

  const handleFileChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    const files = Array.from(e.target.files || []);
    const total = imageFiles.length + files.length;
    if (total > 3) {
      alert('单次最多上传 3 张图片');
      return;
    }
    setImageFiles((prev) => [...prev, ...files]);
    e.target.value = '';
  };

  const removeImage = (i: number) => {
    setImageFiles((prev) => prev.filter((_, idx) => idx !== i));
  };

  return (
    <div style={{ animationDelay: '0.2s' }} className="flex-1 flex flex-col min-w-0 min-h-0 glass rounded-2xl border border-border shadow-card overflow-hidden animate-rise">
      {/* 消息列表 */}
      <div className="flex-1 overflow-y-auto px-4 py-4 space-y-4">
        {messages.length === 0 ? (
          <div className="flex items-center justify-center h-full">
            <div className="relative text-center animate-rise">
              <div
                className="pointer-events-none absolute left-1/2 top-2 -z-10 h-56 w-56 -translate-x-1/2 rounded-full bg-primary/25 blur-3xl"
                aria-hidden="true"
              />
              <ZagentLogo size={72} className="mx-auto mb-5 rounded-3xl shadow-glow animate-floaty" />
              <h2 className="text-3xl font-extrabold tracking-tight text-gradient animate-gradient">
                Zagent
              </h2>
              <p className="mt-2 text-sm text-fg-muted">
                {sessionId ? `会话 ${sessionId}` : 'AI Agent 编排控制台'}
              </p>
              <p className="mt-0.5 text-xs text-fg-subtle">输入问题开始对话</p>
              <div className="mt-5 flex justify-center gap-2 text-[11px]">
                {['ReAct', 'Plan', 'Multi-Agent'].map((t) => (
                  <span
                    key={t}
                    className="rounded-full border border-primary/25 bg-primary/10 text-accent px-2.5 py-1 font-medium"
                  >
                    {t}
                  </span>
                ))}
              </div>
            </div>
          </div>
        ) : (
          messages.map((m, i) => (
            <MessageBubble
              key={m.id}
              message={m}
              streaming={loading && i === messages.length - 1 && m.role === 'assistant' && m.content.length > 0}
            />
          ))
        )}

        {loading && !messages[messages.length - 1]?.content && (
          <div className="flex items-center gap-2.5 pl-11 animate-fade-in-up">
            <span className="loading-dots flex items-center" aria-hidden="true">
              <span /><span /><span />
            </span>
            <span className="text-sm text-fg-muted">Zagent 思考中...</span>
          </div>
        )}

        {error && (
          <div className="bg-danger/10 border border-danger/20 text-danger text-xs px-3 py-2 rounded-lg">
            {error}
          </div>
        )}

        <div ref={messagesEnd} />
      </div>

      {/* 输入区 */}
      <div className="border-t border-border bg-surface/40 px-4 py-3">
        {/* 已选图片预览 */}
        {imageFiles.length > 0 && (
          <div className="flex flex-wrap gap-1.5 mb-2.5">
            {imageFiles.map((f, i) => (
              <div key={i} className="flex items-center gap-1 text-[11px] bg-accent/10 text-accent px-2 py-1 rounded-md border border-accent/20">
                <ImagePlus size={11} />
                <span className="max-w-[80px] truncate">{f.name}</span>
                <button onClick={() => removeImage(i)} className="hover:text-danger">
                  <X size={11} />
                </button>
              </div>
            ))}
          </div>
        )}

        <div className="flex items-end gap-2">
          {/* 图片上传 */}
          <label className="shrink-0 p-2.5 rounded-xl border border-border text-fg-subtle hover:text-primary hover:border-primary/40 cursor-pointer transition-all bg-surface-2 lift active:scale-90">
            <ImagePlus size={16} />
            <input
              type="file"
              accept="image/*"
              multiple
              onChange={handleFileChange}
              className="hidden"
            />
          </label>

          {/* 文本输入 */}
          <textarea
            ref={textareaRef}
            value={input}
            onChange={(e) => setInput(e.target.value)}
            onKeyDown={handleKeyDown}
            placeholder="输入问题，Enter 发送，Shift+Enter 换行..."
            rows={1}
            disabled={loading}
            className="flex-1 resize-none text-sm px-3.5 py-2.5 border border-border rounded-xl
                       bg-surface-2 text-fg focus:outline-none focus:border-primary focus:ring-2
                       focus:ring-ring/30 placeholder:text-fg-subtle disabled:opacity-50
                       transition-colors min-h-[40px] max-h-[120px]"
          />

          {/* 发送 */}
          <button
            onClick={handleSend}
            disabled={loading || !input.trim()}
            className="shrink-0 p-2.5 rounded-xl bg-brand-gradient text-white shadow-glow lift active:scale-90
                       disabled:opacity-30 disabled:cursor-not-allowed disabled:shadow-none transition-all"
          >
            {loading ? (
              <Loader2 size={16} className="animate-spin" />
            ) : (
              <Send size={16} />
            )}
          </button>
        </div>

        {/* 底部操作 */}
        <div className="flex items-center justify-between mt-2">
          <div className="flex items-center gap-2">
            {/* 执行模式选择器（Phase 2）——点击展开 */}
            <div className="relative" ref={modeRef}>
              <button
                onClick={() => setModeMenuOpen(!modeMenuOpen)}
                className="flex items-center gap-1 text-[10px] text-fg-subtle hover:text-primary transition-colors px-2 py-1 rounded-lg hover:bg-accent/10 border border-border"
                title={MODE_OPTIONS.find(o => o.value === execMode)?.desc}
              >
                <span className="text-fg-muted font-medium">
                  {MODE_OPTIONS.find(o => o.value === execMode)?.label || '自动'}
                </span>
                <ChevronDown size={9} className={`transition-transform ${modeMenuOpen ? 'rotate-180' : ''}`} />
              </button>
              {/* 下拉菜单——点击显示，无间隙 */}
              {modeMenuOpen && (
                <div className="absolute bottom-full left-0 mb-1 flex flex-col glass border border-border rounded-xl shadow-pop z-10 min-w-[170px] py-1 animate-rise">
                  {MODE_OPTIONS.map((opt) => (
                    <button
                      key={opt.value}
                      onClick={() => { onModeChange(opt.value); setModeMenuOpen(false); }}
                      className={`text-left px-3 py-1.5 text-xs transition-colors ${
                        execMode === opt.value
                          ? 'bg-accent/10 text-accent font-medium'
                          : 'text-fg-muted hover:bg-surface-2'
                      }`}
                    >
                      <div className="font-medium">{opt.label}</div>
                      <div className="text-[10px] text-fg-subtle leading-tight">{opt.desc}</div>
                    </button>
                  ))}
                </div>
              )}
            </div>

            {sessionId && (
              <button
                onClick={onEndSession}
                className="flex items-center gap-1 text-[10px] text-fg-subtle hover:text-danger transition-colors px-2 py-1 rounded-lg hover:bg-danger/10"
              >
                <Square size={10} />
                结束会话
              </button>
            )}
            {messages.length > 0 && (
              <button
                onClick={onClearMessages}
                className="flex items-center gap-1 text-[10px] text-fg-subtle hover:text-fg transition-colors px-2 py-1 rounded-lg hover:bg-surface-2"
              >
                <Trash2 size={10} />
                清空界面
              </button>
            )}
          </div>

          {sessionId && (
            <span className="text-[10px] text-fg-subtle font-mono">
              {sessionId}
            </span>
          )}
        </div>
      </div>
    </div>
  );
}
