import { useState, useRef } from 'react';
import { Upload, FileText, CheckCircle, AlertCircle, Loader2, Database, ChevronRight } from 'lucide-react';
import { uploadKnowledge } from '../lib/api';

const EXAMPLE_FILES: any[] = [];

interface Props {
  onUploaded: () => void;
}

export default function KnowledgePanel({ onUploaded }: Props) {
  const [uploading, setUploading] = useState(false);
  const [result, setResult] = useState<{ ok: boolean; msg: string } | null>(null);
  const [collapsed, setCollapsed] = useState(true);
  const fileRef = useRef<HTMLInputElement>(null);

  const handleUpload = async (file: File) => {
    setUploading(true);
    setResult(null);
    try {
      const res = await uploadKnowledge(file);
      setResult({ ok: true, msg: `${res.filename}: ${res.msg}` });
      onUploaded();
    } catch (e: any) {
      setResult({ ok: false, msg: e.message || '上传失败' });
    } finally {
      setUploading(false);
    }
  };

  const handleFileChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (file) handleUpload(file);
    e.target.value = '';
  };

  // 折叠态：竖排图标条
  if (collapsed) {
    return (
      <aside className="w-11 shrink-0 glass border border-border rounded-2xl shadow-card flex flex-col items-center py-3 gap-3 animate-rise">
        <button
          onClick={() => setCollapsed(false)}
          className="p-1.5 rounded-lg text-primary hover:bg-surface-2 transition-transform lift active:scale-90"
          title="展开知识库"
          aria-label="展开知识库"
        >
          <Database size={16} />
        </button>
        <span className="text-[10px] tracking-widest text-fg-subtle [writing-mode:vertical-rl] select-none">
          知识库
        </span>
        {result && (
          <span
            className={`w-1.5 h-1.5 rounded-full ${result.ok ? 'bg-success' : 'bg-danger'}`}
            title={result.msg}
          />
        )}
      </aside>
    );
  }

  // 展开态：完整侧栏
  return (
    <aside className="w-64 shrink-0 glass border border-border rounded-2xl shadow-card flex flex-col overflow-hidden animate-rise">
      <header className="flex items-center gap-2 px-3.5 py-3 border-b border-border">
        <Database size={15} className="text-primary" />
        <h3 className="text-sm font-semibold text-fg flex-1">知识库管理</h3>
        <button
          onClick={() => setCollapsed(true)}
          className="p-1 rounded-lg text-fg-subtle hover:text-fg hover:bg-surface-2 transition-transform active:scale-90"
          title="折叠"
          aria-label="折叠知识库"
        >
          <ChevronRight size={15} />
        </button>
      </header>

      <div className="p-3.5 flex-1 overflow-y-auto space-y-3">
        {/* 上传区 */}
        <label className={`flex items-center justify-center gap-2 w-full px-4 py-2.5 rounded-xl border text-sm font-medium cursor-pointer transition-all lift active:scale-[0.98]
          ${uploading
            ? 'bg-surface-2 border-border text-fg-subtle'
            : 'bg-brand-gradient text-white border-transparent shadow-glow'
          }`}>
          {uploading ? (
            <Loader2 size={15} className="animate-spin" />
          ) : (
            <Upload size={15} />
          )}
          {uploading ? '上传中...' : '上传文档'}
          <input
            ref={fileRef}
            type="file"
            accept=".md,.markdown,.txt,.csv,.json,.pdf,.docx,.doc,.pptx,.ppt,.xlsx,.xls"
            onChange={handleFileChange}
            disabled={uploading}
            className="hidden"
          />
        </label>

        {/* 结果 */}
        {result && (
          <div className={`flex items-start gap-1.5 text-xs px-3 py-1.5 rounded-md ${
            result.ok
              ? 'bg-success/10 text-success border border-success/20'
              : 'bg-danger/10 text-danger border border-danger/20'
          }`}>
            {result.ok ? <CheckCircle size={12} className="mt-0.5 shrink-0" /> : <AlertCircle size={12} className="mt-0.5 shrink-0" />}
            <span className="min-w-0 break-words">{result.msg}</span>
          </div>
        )}

        {/* 示例文件 */}
        <div>
          <p className="text-[10px] text-fg-subtle mb-1.5">知识库已有文档：</p>
          <div className="flex flex-col gap-1.5">
            {EXAMPLE_FILES.map((f) => (
              <span key={f} className="inline-flex items-center gap-1.5 text-[11px] text-fg-muted bg-surface-2 px-2 py-1 rounded-md border border-border">
                <FileText size={11} className="text-accent shrink-0" />
                <span className="truncate">{f}</span>
              </span>
            ))}
          </div>
        </div>
      </div>
    </aside>
  );
}
