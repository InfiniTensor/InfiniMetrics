/**
 * 界面展示用日期统一为 YYYY-MM-DD（ISO 年月日）。
 * 支持：YYYY-MM-DD、YYYY/MM/DD、YYYYMMDD、可选的 ISO 日期时间前缀、美式 M/D/YY 与 M/D/YYYY。
 */
export function formatDisplayDateYmd(input: string | undefined | null): string {
  const raw = String(input ?? '').trim()
  if (!raw) return ''

  const isoHead = /^(\d{4})-(\d{2})-(\d{2})/.exec(raw)
  if (isoHead) return `${isoHead[1]}-${isoHead[2]}-${isoHead[3]}`

  const ymdSep = /^(\d{4})[-/](\d{1,2})[-/](\d{1,2})$/.exec(raw)
  if (ymdSep) {
    const y = ymdSep[1]
    const mo = ymdSep[2]!.padStart(2, '0')
    const d = ymdSep[3]!.padStart(2, '0')
    return `${y}-${mo}-${d}`
  }

  const compact = /^(\d{4})(\d{2})(\d{2})$/.exec(raw)
  if (compact) return `${compact[1]}-${compact[2]}-${compact[3]}`

  // 美式 M/D/YY 或 M/D/YYYY（如元数据 bwDatasetUpdatedAt: "4/29/26"）
  const us = /^(\d{1,2})\/(\d{1,2})\/(\d{2}|\d{4})$/.exec(raw)
  if (us) {
    const month = Number(us[1])
    const day = Number(us[2])
    let year = Number(us[3])
    if (us[3]!.length === 2) year += year < 70 ? 2000 : 1900
    if (
      month >= 1 &&
      month <= 12 &&
      day >= 1 &&
      day <= 31 &&
      year >= 1900 &&
      year <= 9999
    ) {
      return `${year}-${String(month).padStart(2, '0')}-${String(day).padStart(2, '0')}`
    }
  }

  return raw
}
