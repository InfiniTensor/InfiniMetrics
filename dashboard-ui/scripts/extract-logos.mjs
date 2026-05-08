import fs from 'fs'
import { fileURLToPath } from 'url'
import { dirname, join } from 'path'
const __dirname = dirname(fileURLToPath(import.meta.url))
const html = fs.readFileSync(join(__dirname, '../../dashboard_preview.html'), 'utf8')
const re = /<img src="(data:image[^"]+)"/g
let m
let i = 0
const out = []
while ((m = re.exec(html)) !== null) {
  out.push(`export const LOGO_${i === 0 ? 'JIUYUAN' : 'INFINITENSOR'} = ${JSON.stringify(m[1])}`)
  i++
}
const outDir = join(__dirname, '../src/data')
fs.mkdirSync(outDir, { recursive: true })
fs.writeFileSync(join(outDir, 'logoDataUri.ts'), out.join('\n') + '\n')
console.log('wrote', i, 'logos')
