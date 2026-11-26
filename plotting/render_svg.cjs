const fs = require('node:fs');
const sharp = require('sharp');
async function main() {
  const [input, output, dpiText, widthText, heightText] = process.argv.slice(2);
  const dpi = Number(dpiText || 600);
  const svg = fs.readFileSync(input, 'utf8');
  const width = widthText ? Number(widthText) : Math.round(Number(svg.match(/<svg[^>]*\bwidth="([\d.]+)pt"/)[1]) * dpi / 72);
  const height = heightText ? Number(heightText) : Math.round(Number(svg.match(/<svg[^>]*\bheight="([\d.]+)pt"/)[1]) * dpi / 72);
  const viewport = svg.replace(/(<svg[^>]*\bwidth=")[\d.]+pt"/, `$1${width}px"`)
    .replace(/(<svg[^>]*\bheight=")[\d.]+pt"/, `$1${height}px"`);
  await sharp(Buffer.from(viewport), {density: 72})
    .flatten({background:'#ffffff'}).withMetadata({density:dpi}).png().toFile(output);
}
main().catch(e => {process.stderr.write(String(e));process.exitCode=1;});
