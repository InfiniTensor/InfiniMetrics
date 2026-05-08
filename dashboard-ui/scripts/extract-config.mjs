import fs from 'fs'
import { fileURLToPath } from 'url'
import { dirname, join } from 'path'
const __dirname = dirname(fileURLToPath(import.meta.url))
const html = fs.readFileSync(join(__dirname, '../../dashboard_preview.html'), 'utf8')
const start = html.indexOf('const PLATFORMS =')
const end = html.indexOf('let selectedPlats', start)
if (start < 0 || end < 0) throw new Error('markers not found')
let block = html.slice(start, end).trim()
block = block.replace(/^const /gm, 'export const ')
// 删除 STATE 之后无关 - we only took until STATE comment

const header = `/** 自 dashboard_preview.html 同步的静态配置，勿改业务含义 */
`
const outDir = join(__dirname, '../src/data')
fs.mkdirSync(outDir, { recursive: true })
fs.writeFileSync(join(outDir, 'dashboardConfig.ts'), header + block + '\n')
console.log('wrote dashboardConfig.ts, len', block.length)
