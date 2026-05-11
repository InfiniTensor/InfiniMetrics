/** Vue Router params 可能是 `string | string[]`，统一成单个字符串 */
export function routeParamString(v: unknown): string {
  if (v == null) return ''
  if (Array.isArray(v)) return String(v[0] ?? '').trim()
  return String(v).trim()
}
