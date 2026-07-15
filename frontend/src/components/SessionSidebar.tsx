import { useState } from 'react';
import {
  Users, RefreshCw, ChevronRight, User, Calendar, MessageSquare,
  Sparkles, Smartphone, AlertCircle, Plus, Trash2,
} from 'lucide-react';
import type { SessionSummary, UserProfile } from '../types';

interface Props {
  userId: string;
  onUserIdChange: (id: string) => void;
  sessions: SessionSummary[];
  sessionsLoading: boolean;
  onRefreshSessions: () => void;
  currentSessionId: string | null;
  onSelectSession: (id: string) => void;
  onNewSession: () => void;
  onDeleteSession: (id: string) => void;
  profile: UserProfile | null;
  profileLoading: boolean;
}

export default function SessionSidebar({
  userId,
  onUserIdChange,
  sessions,
  sessionsLoading,
  onRefreshSessions,
  currentSessionId,
  onSelectSession,
  onNewSession,
  onDeleteSession,
  profile,
  profileLoading,
}: Props) {
  const [inputVal, setInputVal] = useState(userId);

  const handleBlur = () => {
    if (inputVal.trim()) onUserIdChange(inputVal.trim());
  };

  const handleKeyDown = (e: React.KeyboardEvent) => {
    if (e.key === 'Enter' && inputVal.trim()) {
      onUserIdChange(inputVal.trim());
    }
  };

  const profileData = profile?.data;
  const hasProfile = profileData && (Object.keys(profileData.profile).length > 0 || profileData.preferences.length > 0);

  return (
    <aside style={{ animationDelay: '0.08s' }} className="w-64 shrink-0 glass border border-border rounded-2xl shadow-card flex flex-col min-h-0 overflow-hidden animate-rise">
      {/* 用户 ID */}
      <div className="p-3 border-b border-border">
        <label className="text-[10px] uppercase tracking-widest text-fg-subtle mb-1 block font-medium">
          用户 ID
        </label>
        <div className="flex gap-1.5">
          <input
            value={inputVal}
            onChange={(e) => setInputVal(e.target.value)}
            onBlur={handleBlur}
            onKeyDown={handleKeyDown}
            className="flex-1 text-xs px-2.5 py-1.5 border border-border rounded-md bg-surface-2
                       focus:outline-none focus:border-primary focus:ring-2 focus:ring-ring/30
                       font-mono text-fg transition-colors"
            placeholder="default"
          />
        </div>
      </div>

      {/* 活跃会话 */}
      <div className="p-3 border-b border-border flex-1 overflow-hidden flex flex-col min-h-0">
        <div className="flex items-center justify-between mb-2">
          <label className="text-[10px] uppercase tracking-widest text-fg-subtle font-medium flex items-center gap-1">
            <Users size={11} />
            活跃会话
          </label>
          <div className="flex items-center gap-1">
            <button
              onClick={onNewSession}
              className="text-fg-subtle hover:text-primary transition-colors"
              title="新建会话"
            >
              <Plus size={13} />
            </button>
            <button
              onClick={onRefreshSessions}
              className="text-fg-subtle hover:text-fg transition-colors"
              title="刷新"
            >
              <RefreshCw size={12} className={sessionsLoading ? 'animate-spin' : ''} />
            </button>
          </div>
        </div>

        <div className="flex-1 overflow-y-auto -mx-1 px-1 space-y-0.5">
          {sessionsLoading && sessions.length === 0 ? (
            <div className="text-[11px] text-fg-subtle py-4 text-center">加载中...</div>
          ) : sessions.length === 0 ? (
            <div className="text-[11px] text-fg-subtle py-4 text-center">
              暂无会话
              <button onClick={onNewSession} className="ml-1 text-primary hover:underline">新建</button>
            </div>
          ) : (
            sessions.map((s) => (
              <div
                key={s.session_id}
                className={`w-full text-left px-2.5 py-2 rounded-lg text-xs transition-all flex items-center justify-between group lift
                  ${currentSessionId === s.session_id
                    ? 'bg-accent/10 text-accent border border-accent/30 shadow-sm'
                    : 'hover:bg-surface-2 text-fg-muted border border-transparent'
                  }`}
              >
                <button
                  onClick={() => onSelectSession(s.session_id)}
                  className="flex items-center gap-1.5 min-w-0 flex-1"
                >
                  <MessageSquare size={11} className="shrink-0" />
                  <span className="truncate font-mono text-[11px]">{s.session_id}</span>
                </button>
                <div className="flex items-center gap-1 shrink-0 ml-1.5">
                  <span className="text-[10px] text-fg-subtle">{s.turn_count}轮</span>
                  <button
                    onClick={(e) => { e.stopPropagation(); onDeleteSession(s.session_id); }}
                    className="text-fg-subtle hover:text-danger opacity-0 group-hover:opacity-100 transition-all"
                    title="删除会话"
                  >
                    <Trash2 size={11} />
                  </button>
                </div>
              </div>
            ))
          )}
        </div>
      </div>

      {/* 用户画像 */}
      <div className="p-3 border-t border-border">
        <label className="text-[10px] uppercase tracking-widest text-fg-subtle mb-2 block font-medium flex items-center gap-1">
          <User size={11} />
          用户画像
        </label>

        {profileLoading ? (
          <div className="text-[11px] text-fg-subtle py-2">加载中...</div>
        ) : !hasProfile ? (
          <div className="text-[11px] text-fg-subtle py-2">暂无画像数据</div>
        ) : (
          <div className="space-y-2 text-[11px]">
            {Object.keys(profileData.profile).length > 0 && (
              <div>
                <div className="text-[10px] text-fg-subtle mb-0.5 font-medium">属性</div>
                {Object.entries(profileData.profile).map(([k, v]) => (
                  <div key={k} className="flex justify-between text-[11px] py-0.5">
                    <span className="text-fg-muted">{k}</span>
                    <span className="text-fg font-medium">{v}</span>
                  </div>
                ))}
              </div>
            )}
            {profileData.preferences.length > 0 && (
              <div>
                <div className="text-[10px] text-fg-subtle mb-1 font-medium flex items-center gap-1">
                  <Sparkles size={10} />偏好
                </div>
                <div className="flex flex-wrap gap-1">
                  {profileData.preferences.map((p, i) => (
                    <span key={i} className="px-1.5 py-0.5 bg-warning/10 text-warning rounded text-[10px]">
                      {p}
                    </span>
                  ))}
                </div>
              </div>
            )}
            {profileData.mentioned_products.length > 0 && (
              <div>
                <div className="text-[10px] text-fg-subtle mb-1 font-medium flex items-center gap-1">
                  <Smartphone size={10} />关注产品
                </div>
                <div className="flex flex-wrap gap-1">
                  {profileData.mentioned_products.map((p, i) => (
                    <span key={i} className="px-1.5 py-0.5 bg-accent/10 text-accent rounded text-[10px]">
                      {p}
                    </span>
                  ))}
                </div>
              </div>
            )}
            <div className="text-[10px] text-fg-subtle flex items-center gap-1">
              <Calendar size={10} />
              互动 {profileData.interaction_count} 次
            </div>
          </div>
        )}
      </div>
    </aside>
  );
}
