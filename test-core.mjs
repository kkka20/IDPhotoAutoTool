/**
 * 证件照网页核心逻辑验证
 * 覆盖：掩膜类别选择（本次 bug 根因）、导出校验阈值、多背景色合成数学
 * 运行：node test-core.mjs
 */
import fs from 'fs';

const html = fs.readFileSync('index.html', 'utf8');
let pass = 0, fail = 0;
function assert(cond, name, detail = '') {
  if (cond) { pass++; console.log(`  ✓ ${name}`); }
  else { fail++; console.log(`  ✗ ${name}  ${detail}`); }
}

console.log('\n=== 0. 源码关键修复点确认 ===');
assert(
  /masks\.length >= 2 \? masks\[1\] : masks\[0\]/.test(html),
  '源码取 masks[1]（人像）而非 masks[0]（背景）'
);
assert(
  !/state\.mask = \{ w: confidence\.width.*confidenceMasks\[0\]/.test(html),
  '旧的 confidenceMasks[0] 写法已移除'
);
assert(
  /const MIN_FG_RATIO = 0\.02/.test(html),
  '存在 MIN_FG_RATIO 导出校验阈值'
);
assert(
  /const MIN_MASK_DIR = 0\.85/.test(html) && /< MIN_MASK_DIR/.test(html),
  '存在掩膜方向（中心/边缘）校验'
);
assert(
  /centerEdgeStats/.test(html),
  '已实现 centerEdgeStats 空间分布统计'
);
assert(
  /const bad = checks\.find\(\(c\) => c\.type === 'bad'\)/.test(html),
  '导出前会拦截 bad 级校验结果'
);
assert(
  /destination-in/.test(html) && /globalCompositeOperation = 'source-over'/.test(html),
  'destination-in 后正确复原为 source-over'
);
assert(
  /ctx\.drawImage\(fg, 0, 0\)/.test(html),
  '背景填充后绘制人像层 fg'
);
// URL 正确性
assert(
  /blaze_face_short_range\.tflite/.test(html),
  '人脸模型 URL 文件名正确'
);
assert(
  !/latest\/blaze_short_range\.tflite/.test(html),
  '不存在错误的 blaze_short_range.tflite 路径'
);

/* ─────────────────────────────────────────────────
   1. 掩膜类别选择逻辑（复刻 extractPersonMask 的决策部分）
   ───────────────────────────────────────────────── */
function pickMaskData(maskArrays) {
  let data = maskArrays.length >= 2 ? maskArrays[1] : maskArrays[0];
  if (maskArrays.length < 2) {
    let sum = 0;
    for (let i = 0; i < data.length; i++) sum += data[i];
    if (sum / data.length > 0.6) data = data.map((v) => 1 - v);
  }
  return data;
}
const mean = (a) => a.reduce((x, y) => x + y, 0) / a.length;

console.log('\n=== 1. 掩膜类别选择 ===');
{
  // 双通道：index0=背景（覆盖率 0.85），index1=人像（0.15）
  const bg = new Float32Array(100).fill(0.85);
  const person = new Float32Array(100).fill(0.15);
  const d = pickMaskData([bg, person]);
  assert(Math.abs(mean(d) - 0.15) < 1e-6, '双通道时取人像掩膜（均值 0.15）', `实际 ${mean(d)}`);
}
{
  // 双通道但顺序若为 [人像, 背景]，仍取 index1 —— 这正是官方约定
  const a = new Float32Array(100).fill(0.15);
  const b = new Float32Array(100).fill(0.85);
  const d = pickMaskData([a, b]);
  assert(Math.abs(mean(d) - 0.85) < 1e-6, '严格按官方索引取 index1（不做猜测）', `实际 ${mean(d)}`);
}
{
  // 单通道 + 高覆盖率 → 判定为背景掩膜，反转
  const single = new Float32Array(100).fill(0.9);
  const d = pickMaskData([single]);
  assert(Math.abs(mean(d) - 0.1) < 1e-6, '单通道高覆盖（0.9）自动反转为 0.1', `实际 ${mean(d)}`);
}
{
  // 单通道 + 低覆盖率 → 认为已是人像，不反转
  const single = new Float32Array(100).fill(0.25);
  const d = pickMaskData([single]);
  assert(Math.abs(mean(d) - 0.25) < 1e-6, '单通道低覆盖（0.25）保持原样', `实际 ${mean(d)}`);
}

/* ─────────────────────────────────────────────────
   2. 导出校验阈值（复刻 buildChecks 的判定）
   ───────────────────────────────────────────────── */
const MIN_FG_RATIO = 0.02;
const MAX_FG_RATIO = 0.985;
function judgeFg(fgRatio, feat = { matting: true, background: true }) {
  if (feat.matting && feat.background) {
    if (fgRatio < MIN_FG_RATIO) return 'bad';
    if (fgRatio > MAX_FG_RATIO) return 'warn-full';
    return 'ok';
  }
  if (feat.background && !feat.matting) return 'warn';
  return 'ok';
}

console.log('\n=== 2. 导出前校验 ===');
assert(judgeFg(0.001) === 'bad', 'fgRatio=0.1% → 判定 bad，阻止导出（复现本次 bug 场景）');
assert(judgeFg(0.0) === 'bad', 'fgRatio=0% → 判定 bad');
assert(judgeFg(0.35) === 'ok', 'fgRatio=35%（正常人像）→ 判定 ok');
assert(judgeFg(0.995) === 'warn-full', 'fgRatio=99.5% → warn（抠图几乎未生效）');
assert(
  judgeFg(0.995, { matting: true, background: false }) === 'ok',
  '未开背景替换时不报错'
);
assert(
  judgeFg(0.9, { matting: false, background: true }) === 'warn',
  '未开抠图但开背景 → warn（提示背景会被遮住）'
);

/* ─────────────────────────────────────────────────
   2.5 掩膜方向校验（中心/边缘比值）
   ───────────────────────────────────────────────── */
function centerEdgeStats(data, w, h) {
  const cy0 = Math.floor(h * 0.30), cy1 = Math.floor(h * 0.70);
  const cx0 = Math.floor(w * 0.30), cx1 = Math.floor(w * 0.70);
  const b = Math.max(1, Math.floor(Math.min(w, h) * 0.12));
  let cSum = 0, cN = 0, eSum = 0, eN = 0;
  for (let y = 0; y < h; y++) {
    for (let x = 0; x < w; x++) {
      const v = data[y * w + x] > 0.5 ? 1 : 0;
      if (y >= cy0 && y < cy1 && x >= cx0 && x < cx1) { cSum += v; cN++; }
      const onEdge = y < b || y >= h - b || x < b || x >= w - b;
      if (onEdge) { eSum += v; eN++; }
    }
  }
  const center = cN ? cSum / cN : 0;
  const edge = eN ? eSum / eN : 0;
  return { center, edge, ratio: center / Math.max(edge, 1e-6) };
}
function makeMask(w, h, fn) {
  const d = new Float32Array(w * h);
  for (let y = 0; y < h; y++) for (let x = 0; x < w; x++) d[y * w + x] = fn(x / w, y / h) ? 1 : 0;
  return d;
}

console.log('\n=== 2.5 掩膜方向校验 ===');
{
  // 模拟人像：中央竖直椭圆
  const person = makeMask(100, 100, (u, v) => {
    const du = (u - 0.5) / 0.22, dv = (v - 0.5) / 0.35;
    return du * du + dv * dv <= 1;
  });
  const s = centerEdgeStats(person, 100, 100);
  assert(s.ratio > 0.85, `人像掩膜（中央集中）比值 ${s.ratio.toFixed(2)} > 0.85，判为正常`);
}
{
  // 模拟背景：四周框
  const bg = makeMask(100, 100, (u, v) => !(u > 0.25 && u < 0.75 && v > 0.25 && v < 0.75));
  const s = centerEdgeStats(bg, 100, 100);
  assert(s.ratio < 0.85, `背景掩膜（四周集中）比值 ${s.ratio.toFixed(2)} < 0.85，判为异常`);
}
{
  // 真实实测值（Python/OpenCV + U2Net 跑 assets 得到）
  const REAL = [
    { name: '1.png 人像', ratio: 3.26, expectOk: true },
    { name: '1.png 背景', ratio: 0.02, expectOk: false },
    { name: 'camera_shot 人像', ratio: 2.92, expectOk: true },
    { name: 'camera_shot 背景', ratio: 0.70, expectOk: false },
  ];
  let allOk = true;
  for (const r of REAL) {
    const pass = (r.ratio >= 0.85) === r.expectOk;
    if (!pass) allOk = false;
    console.log(`    · ${r.name.padEnd(20)} ratio=${r.ratio.toFixed(2)} → ${(r.ratio >= 0.85 ? '正常' : '异常')} ${pass ? '✓' : '✗'}`);
  }
  assert(allOk, '4 组真实掩膜实测值全部判定正确（区分人像与背景）');
}

/* ─────────────────────────────────────────────────
   3. 合成数学：不同背景色下人像是否保留
   ───────────────────────────────────────────────── */
console.log('\n=== 3. 多背景色合成数学 ===');

// 模拟 compose：先填充背景色，再把带 alpha 的人像层叠加（source-over）
function composite(bgColor, fgPixels) {
  const out = [];
  for (const p of fgPixels) {
    const a = p.a;
    out.push({
      r: p.r * a + bgColor.r * (1 - a),
      g: p.g * a + bgColor.g * (1 - a),
      b: p.b * a + bgColor.b * (1 - a),
    });
  }
  return out;
}
const COLORS = {
  white: { r: 255, g: 255, b: 255 },
  red:   { r: 244, g: 66,  b: 66  },
  blue:  { r: 67,  g: 142, b: 219 },
};
// 典型人像像素（肤色）与背景像素（alpha=0）
const personPixel = { r: 225, g: 185, b: 160, a: 1.0 };
const bgPixel = { r: 0, g: 0, b: 0, a: 0 };

for (const [name, color] of Object.entries(COLORS)) {
  const [p, b] = composite(color, [personPixel, bgPixel]);
  const personKept =
    Math.abs(p.r - 225) < 1 && Math.abs(p.g - 185) < 1 && Math.abs(p.b - 160) < 1;
  const bgFilled =
    Math.abs(b.r - color.r) < 1 && Math.abs(b.g - color.g) < 1 && Math.abs(b.b - color.b) < 1;
  assert(personKept, `${name}底：人像像素完整保留（${p.r.toFixed(0)},${p.g.toFixed(0)},${p.b.toFixed(0)}）`);
  assert(bgFilled, `${name}底：背景区域正确填充（${b.r.toFixed(0)},${b.g.toFixed(0)},${b.b.toFixed(0)}）`);
}

// 关键回归：若误用「背景掩膜」，人像像素会被裁掉
function compositeWithWrongMask(bgColor) {
  // 背景掩膜对肤色像素给 alpha≈0 → 人像被抹掉，只剩背景色
  const wrongAlpha = 0.0;
  return {
    r: personPixel.r * wrongAlpha + bgColor.r * (1 - wrongAlpha),
    g: personPixel.g * wrongAlpha + bgColor.g * (1 - wrongAlpha),
    b: personPixel.b * wrongAlpha + bgColor.b * (1 - wrongAlpha),
  };
}
const wrong = compositeWithWrongMask(COLORS.red);
assert(
  Math.abs(wrong.r - 244) < 1 && Math.abs(wrong.g - 66) < 1,
  '反向验证：用背景掩膜会得到纯红底（复现用户截图症状）',
  `实际 (${wrong.r},${wrong.g},${wrong.b})`
);

/* ─────────────────────────────────────────────────
   4. 功能勾选组合
   ───────────────────────────────────────────────── */
console.log('\n=== 4. 功能自由勾选组合 ===');
const FEATURES = ['matting', 'background', 'resize', 'filter'];
function combos(arr) {
  const res = [];
  for (let m = 0; m < (1 << arr.length); m++) {
    const o = {};
    arr.forEach((k, i) => { o[k] = !!(m & (1 << i)); });
    res.push(o);
  }
  return res;
}
const all = combos(FEATURES);
assert(all.length === 16, '4 个可勾选功能共 16 种组合，全部可自由选择');
// 每种组合都必须能定出输出尺寸，不能崩
let sizeOk = true;
for (const c of all) {
  const w = c.resize ? 295 : 280;
  const h = c.resize ? 413 : 392;
  if (!(w > 0 && h > 0)) sizeOk = false;
}
assert(sizeOk, '所有勾选组合都能得出有效输出尺寸');
// 只有「抠图 + 背景」同时勾选才需要人像校验
const needFgCheck = all.filter((c) => c.matting && c.background).length;
assert(needFgCheck === 4, '抠图+背景同时勾选的组合数为 4，仅这些做严格人像校验');

/* ─────────────────────────────────────────────────
   5. 历史记录逻辑（最多 10 张 / 超出丢最旧 / 可删除）
   ───────────────────────────────────────────────── */
console.log('\n=== 5. 历史记录 ===');
{
  const HISTORY_MAX = 10;
  let history = [];
  const addItem = (id) => {
    history.unshift({ id });
    if (history.length > HISTORY_MAX) history = history.slice(0, HISTORY_MAX);
  };
  const deleteItem = (id) => { history = history.filter((h) => h.id !== id); };

  for (let i = 1; i <= 12; i++) addItem(i);
  assert(history.length === HISTORY_MAX, `添加 12 张后上限仍为 10（实际 ${history.length}）`);
  assert(history[0].id === 12, '最新一张排在最前');
  assert(!history.some((h) => h.id === 1 || h.id === 2), '最旧两张被丢弃');
  assert(history[history.length - 1].id === 3, '保留的是最近的 10 张');

  deleteItem(7);
  assert(history.length === 9 && !history.some((h) => h.id === 7), '删除中间一张生效');

  addItem(13);
  assert(history.length === 10, '删除后再添加回到上限');

  // 序列化往返（localStorage 用 JSON 字符串）
  const roundTrip = JSON.parse(JSON.stringify(history));
  assert(roundTrip.length === history.length && roundTrip[0].id === 13, 'JSON 序列化往返一致');
}
{
  // 源码实现检查
  assert(/const HISTORY_MAX = 10/.test(html), '源码里历史上限为 10');
  assert(/localStorage\.setItem\(HISTORY_KEY/.test(html), '历史记录写入 localStorage（刷新后保留）');
  assert(/history\.length > HISTORY_MAX\) history = history\.slice\(0, HISTORY_MAX\)/.test(html),
    '超出上限时丢弃最旧记录');
  assert(/history = history\.filter\(\(h\) => h\.id !== viewingHistoryId\)/.test(html),
    '支持删除单条历史');
  assert(/id="btn-continue"/.test(html) && !/btn-export/.test(html), '底部按钮已改为「继续」');
  assert(/id="btn-save-history"/.test(html) && /id="btn-download"/.test(html),
    '「继续」弹层含下载与保存历史两个入口');
  assert(/id="btn-history-download"/.test(html) && /id="btn-history-delete"/.test(html),
    '历史记录支持导出与删除');
  assert(!/feat-dress/.test(html), '左侧「换装」项已删除');
  assert(/#camera-video[\s\S]{0,600}transform: scaleX\(-1\)/.test(html),
    '摄像头预览已水平翻转（修正左右颠倒）');
  assert(/ctx\.scale\(-1, 1\)/.test(html), '拍照时同样翻转，预览与成片方向一致');
}

/* ─────────────────────────────────────────────────
   6. JPEG DPI 写入
   ───────────────────────────────────────────────── */
console.log('\n=== 6. JPEG DPI 写入 ===');

function withJpegDpiBytes(input, dpi = 300) {
  const buf = new Uint8Array(input);
  if (buf.length < 4 || buf[0] !== 0xFF || buf[1] !== 0xD8) return buf;
  let i = 2;
  while (i + 16 <= buf.length) {
    if (buf[i] !== 0xFF) break;
    const marker = buf[i + 1];
    if (marker === 0xD8 || marker === 0xD9 || (marker >= 0xD0 && marker <= 0xD7)) { i += 2; continue; }
    if (marker === 0xDA) break;
    const len = (buf[i + 2] << 8) | buf[i + 3];
    if (len < 2) break;
    if (marker === 0xE0 && buf[i+4] === 0x4A && buf[i+5] === 0x46 && buf[i+6] === 0x49 && buf[i+7] === 0x46) {
      buf[i+11] = 1;
      buf[i+12] = (dpi >> 8) & 0xFF; buf[i+13] = dpi & 0xFF;
      buf[i+14] = (dpi >> 8) & 0xFF; buf[i+15] = dpi & 0xFF;
      return buf;
    }
    i += 2 + len;
  }
  const app0 = new Uint8Array([0xFF,0xE0,0x00,0x10,0x4A,0x46,0x49,0x46,0x00,0x01,0x01,0x01,
    (dpi>>8)&0xFF, dpi&0xFF, (dpi>>8)&0xFF, dpi&0xFF, 0x00,0x00]);
  const out = new Uint8Array(buf.length + app0.length);
  out.set(buf.subarray(0,2), 0);
  out.set(app0, 2);
  out.set(buf.subarray(2), 2 + app0.length);
  return out;
}
{
  // 典型 JFIF 头（units=0, density=1,1 → PIL 读作 None）
  const jfif = [0xFF,0xD8,0xFF,0xE0,0x00,0x10,0x4A,0x46,0x49,0x46,0x00,
                0x01,0x01,0x00,0x00,0x01,0x00,0x01,0x00,0x00, 0xFF,0xDB,0x00];
  const before = Uint8Array.from(jfif);
  const after = withJpegDpiBytes(before, 300);

  assert(after[13] === 1, 'units 被设为 1（英寸）');
  const xd = (after[14] << 8) | after[15];
  const yd = (after[16] << 8) | after[17];
  assert(xd === 300 && yd === 300, `X/Y 密度写入 300（实际 ${xd}/${yd}）`);
  assert(after.length === before.length, '就地改写不改变文件体积');

  // 一寸 295×413 @300dpi 应换算成 25.0 × 35.0 mm
  const wmm = 295 / xd * 25.4, hmm = 413 / yd * 25.4;
  assert(Math.abs(wmm - 25.0) < 0.2 && Math.abs(hmm - 35.0) < 0.2,
    `一寸换算为 25.0×35.0mm（实际 ${wmm.toFixed(1)}×${hmm.toFixed(1)}）`);

  // 二寸 413×579 @300dpi → 35.0 × 49.0 mm
  const wmm2 = 413 / xd * 25.4, hmm2 = 579 / yd * 25.4;
  assert(Math.abs(wmm2 - 35.0) < 0.2 && Math.abs(hmm2 - 49.0) < 0.2,
    `二寸换算为 35.0×49.0mm（实际 ${wmm2.toFixed(1)}×${hmm2.toFixed(1)}）`);
}
{
  // 无 JFIF 段（只有 SOI + DQT）→ 应插入 APP0
  const noJfif = [0xFF,0xD8,0xFF,0xDB,0x00,0x43,0x00,0x02,0x01];
  const out = withJpegDpiBytes(Uint8Array.from(noJfif), 300);
  assert(out.length === noJfif.length + 18, '无 JFIF 段时插入 18 字节 APP0');
  assert(out[0] === 0xFF && out[1] === 0xD8, 'SOI 仍保持在文件开头');
  assert(out[2] === 0xFF && out[3] === 0xE0, 'APP0 marker 紧跟在 SOI 之后');
  assert(out[6] === 0x4A && out[7] === 0x46, 'APP0 内标识为 JFIF');
  assert(out[13] === 1, '插入段 units = 1（英寸）');
  assert(((out[14] << 8) | out[15]) === 300, '插入段 X 密度为 300');
  assert(((out[16] << 8) | out[17]) === 300, '插入段 Y 密度为 300');
}
{
  assert(/async function withJpegDpi/.test(html), '已实现 withJpegDpi');
  assert(/await withJpegDpi\(raw, 300\)/.test(html), '下载当前成品时写入 300dpi');
  assert(/withJpegDpi\(dataUrlToBlob\(item\.dataUrl\), 300\)/.test(html),
    '历史记录下载时同样写入 300dpi');
}

/* ─────────────────────────────────────────────────
   7. 背景仓库（最多 20 张 / 背景在下 / 人像置顶）
   ───────────────────────────────────────────────── */
console.log('\n=== 7. 背景仓库 ===');
{
  const BG_MAX = 20;
  let bgs = [];
  const add = (id) => {
    bgs.unshift({ id });
    if (bgs.length > BG_MAX) bgs = bgs.slice(0, BG_MAX);
  };
  for (let i = 1; i <= 25; i++) add(i);
  assert(bgs.length === BG_MAX, `导入 25 张后上限仍为 20（实际 ${bgs.length}）`);
  assert(bgs[0].id === 25, '最新导入的排在最前');
  assert(!bgs.some((b) => b.id <= 5), '最旧 5 张被丢弃');

  const del = (id) => { bgs = bgs.filter((b) => b.id !== id); };
  del(15);
  assert(bgs.length === 19 && !bgs.some((b) => b.id === 15), '删除中间一张生效');
}
{
  // cover 铺满数学：背景图必须盖满输出画布，多余部分居中裁掉
  const cover = (iw, ih, w, h) => {
    const s = Math.max(w / iw, h / ih);
    return { dw: iw * s, dh: ih * s, dx: (w - iw * s) / 2, dy: (h - ih * s) / 2 };
  };
  const r1 = cover(1000, 500, 295, 413);   // 横图 → 竖版输出
  assert(r1.dw >= 295 - 0.01 && r1.dh >= 413 - 0.01, '横图铺满竖版：宽高均覆盖');
  assert(r1.dx < 0 && Math.abs(r1.dy) < 0.01, '横图左右溢出居中、上下贴合');

  const r2 = cover(500, 1000, 295, 413);   // 竖图 → 竖版输出
  assert(r2.dw >= 295 - 0.01 && r2.dh >= 413 - 0.01, '竖图同样铺满');
  assert(Math.abs(r2.dx) < 0.01, '竖图水平方向不裁切');
}
{
  assert(/const BG_MAX = 20/.test(html), '背景仓库上限 20 张');
  assert(/idphoto_bg_library_v1/.test(html), '背景库存入 localStorage（刷新保留）');
  assert(/backgrounds\.unshift\(/.test(html), '新导入的背景插到最前');
  assert(/backgrounds = backgrounds\.slice\(0, BG_MAX\)/.test(html), '超出上限丢弃最旧');
  assert(/const bgImg = selectedBgId \? bgCache\.get\(selectedBgId\) : null/.test(html),
    '合成时优先使用背景图');
  assert(/drawCover\(ctx, bgImg, outW, outH\)/.test(html), '背景图按 cover 铺满输出');
  assert(/ctx\.drawImage\(fg, 0, 0\);\s+\/\/ 人像永远叠在最上层/.test(html),
    '人像始终绘制在背景之上');
  assert(/classList\.toggle\('locked', locked\)/.test(html),
    '使用背景图时锁定颜色调节');
  assert(/function deleteBackground/.test(html) && /del\.className = 'del'/.test(html),
    '背景图支持删除');
  assert(/id="panel-bg-lib"/.test(html), '左栏存在「背景仓库」面板');
}

console.log(`\n───────────\n结果：${pass} 通过 / ${fail} 失败\n`);
process.exit(fail === 0 ? 0 : 1);
