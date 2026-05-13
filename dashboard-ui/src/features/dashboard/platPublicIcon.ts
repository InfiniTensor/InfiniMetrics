/** public 下文件名与 PLATFORMS.key 对应（Vite BASE_URL 兼容子路径部署） */
export const PLAT_ICON_FILE: Record<string, string> = {
  nvidia: 'nvidia.png',
  mthreads: 'moore.png',
  cambricon: 'cambricon.png',
  metax: 'metax.png',
  iluvatar: 'iluvatar.png',
  ascend: 'ascend.png',
  hygon: 'hygon.png',
  generic: 'ali.png',
}

export function platIconSrc(key: string): string {
  const file = PLAT_ICON_FILE[key]
  if (!file) return ''
  const b = import.meta.env.BASE_URL
  return b.endsWith('/') ? `${b}${file}` : `${b}/${file}`
}
