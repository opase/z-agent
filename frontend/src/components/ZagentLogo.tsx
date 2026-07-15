interface Props {
  size?: number;
  className?: string;
}

/** Zagent 品牌标识——圆角方块 + 靛紫渐变底 + "Z" 编排走线 + 起止节点 */
export default function ZagentLogo({ size = 28, className = '' }: Props) {
  return (
    <svg
      width={size}
      height={size}
      viewBox="0 0 32 32"
      fill="none"
      role="img"
      aria-label="Zagent"
      className={className}
    >
      <rect width="32" height="32" rx="8" fill="url(#zagent-grad)" />
      {/* "Z" 走线 = 编排路径 */}
      <path
        d="M10 11 H22 L11 21 H23"
        stroke="#fff"
        strokeWidth="2.2"
        strokeLinecap="round"
        strokeLinejoin="round"
      />
      {/* 起止编排节点 */}
      <circle cx="10" cy="11" r="2.4" fill="#fff" />
      <circle cx="23" cy="21" r="2.4" fill="#fff" />
      <defs>
        <linearGradient
          id="zagent-grad"
          x1="0"
          y1="0"
          x2="32"
          y2="32"
          gradientUnits="userSpaceOnUse"
        >
          <stop stopColor="#6366F1" />
          <stop offset="1" stopColor="#7C3AED" />
        </linearGradient>
      </defs>
    </svg>
  );
}
