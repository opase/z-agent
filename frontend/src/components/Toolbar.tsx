import { useState, useEffect } from 'react';
import { Circle, Sun, Moon } from 'lucide-react';
import type { HealthStatus } from '../types';
import ZagentLogo from './ZagentLogo';

interface Props {
  health: HealthStatus | null;
  healthError: string;
  userId: string;
  onRefresh: () => void;
}

export default function Toolbar({ health, healthError, userId, onRefresh }: Props) {
  const healthy = health?.status === 'healthy';

  // 主题切换（纯 UI 状态，封装于顶栏，不影响任何业务数据流）
  const [theme, setTheme] = useState<'light' | 'dark'>(
    () => (document.documentElement.classList.contains('dark') ? 'dark' : 'light'),
  );
  useEffect(() => {
    document.documentElement.classList.toggle('dark', theme === 'dark');
    localStorage.setItem('zagent-theme', theme);
  }, [theme]);

  return (
    <header className="relative z-20 flex items-center justify-between h-14 px-4 glass border-b border-border shrink-0 select-none animate-fade">
      <div className="flex items-center gap-3">
        <div className="flex items-center gap-2.5">
          <ZagentLogo size={30} className="rounded-xl shadow-glow" />
          <span className="text-base font-bold tracking-tight text-gradient animate-gradient">
            Zagent
          </span>
        </div>
        <span className="text-[10px] text-fg-subtle px-1.5 py-0.5 bg-surface-2 rounded-md font-mono border border-border">
          alpha
        </span>
      </div>

      <div className="flex items-center gap-3">
        {/* 主题切换 */}
        <button
          onClick={() => setTheme((t) => (t === 'dark' ? 'light' : 'dark'))}
          className="p-2 rounded-lg text-fg-muted hover:text-primary hover:bg-surface-2 transition-all lift active:scale-90"
          aria-label={theme === 'dark' ? '切换到浅色主题' : '切换到深色主题'}
          title="切换主题"
        >
          {theme === 'dark' ? <Sun size={16} /> : <Moon size={16} />}
        </button>

        {/* 健康状态 */}
        <button
          onClick={onRefresh}
          className="flex items-center gap-1.5 text-xs text-fg-muted hover:text-fg transition-colors px-2 py-1 rounded-lg hover:bg-surface-2"
          title="刷新状态"
        >
          <Circle
            size={8}
            fill="currentColor"
            stroke="none"
            className={
              healthError || !healthy
                ? 'text-danger'
                : 'text-success animate-pulse-dot'
            }
          />
          <span>
            {healthError
              ? '异常'
              : healthy
                ? `正常 · ${health!.bm25_docs} 条知识 · ${health!.active_sessions} 会话`
                : '加载中...'}
          </span>
        </button>

        {/* 用户 ID */}
        <div className="flex items-center gap-1.5 text-xs text-fg-subtle">
          <span>用户</span>
          <span className="font-mono text-fg-muted bg-surface-2 px-1.5 py-0.5 rounded-md border border-border">
            {userId || 'default'}
          </span>
        </div>
      </div>
    </header>
  );
}
