import fs from 'fs'
import { fileURLToPath } from 'url'
import { dirname, join } from 'path'
const __dirname = dirname(fileURLToPath(import.meta.url))
const html = fs.readFileSync(join(__dirname, '../../dashboard_preview.html'), 'utf8')
const s = html.indexOf('<style>')
const e = html.indexOf('</style>', s)
const css = html.slice(s + 7, e).trim()
const outDir = join(__dirname, '../src/styles')
fs.mkdirSync(outDir, { recursive: true })
fs.writeFileSync(join(outDir, 'dashboard.css'), '/** 与 dashboard_preview.html 样式一致 */\n' + css + '\n')
console.log('wrote dashboard.css', css.length)
