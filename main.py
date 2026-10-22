# -*- coding: utf-8 -*-
"""
项目2：证件照自动裁剪与背景替换工具（v1 —— 核心链路已通）

已实现：
  1. 本地图片加载 / 摄像头实时拍摄（镜像预览 + 保存）
  2. Haar 级联人脸检测 + 多脸最大脸筛选
  3. 由人脸框反推一寸/二寸裁剪框，越界自动补边
  4. 人像抠图（双引擎，见下）
  5. 纯色背景替换（白 / 红 / 蓝 / 任意 BGR）
  6. 按标准证件照尺寸导出（一寸 295×413 / 二寸 413×579，300dpi）

抠图双引擎：
  AI 引擎（默认，推荐）：rembg / U2Net 端到端显著性分割
    - 能保住细发丝，输出原生软 alpha（0~1），不二值化
    - 需装依赖：pip install rembg onnxruntime（首次自动下载 176MB 模型）
    - 单张约 0.2 秒，比 GrabCut 更快
  传统引擎（兜底，纯 OpenCV 无依赖）：GrabCut ∪ 颜色距离
    - 适合离线 / 不能装模型的场景，发丝细节弱于 AI
  用 --engine auto|ai|classic 切换，auto 会在装了 rembg 时选 AI。

抠图之后的三道边缘精修（顺序不能乱）：
  1. refine_alpha_matting —— closed-form Alpha Matting（pymatting）
     U2Net 输出是 320×320 上采样回原尺寸，边界被抹成 5~7px 斜坡；
     实测在纯白背景像素上给出 α=0.18~0.65 的高估值，这是白雾根源。
     Matting 用图像自身颜色在未知带内重解 α，把它压回真实覆盖率。
     依赖 pymatting（rembg 的依赖，pip 装 rembg 时已带上）；缺失则自动跳过。
  2. suppress_background_like —— 背景色先验
     颜色与原背景色几乎相同的像素不可能是人像，强制 α=0。
     只作用于 α<0.95 的过渡带，人像内部高光不受影响。
  3. decontaminate_colors —— 边缘颜色去溢出
     按 I = α·F + (1-α)·B 反解真实前景色 F，剥掉边缘混入的原背景色。
  4. harden_alpha —— 压窄过渡带（α 0.40~0.60 拉满到 0~1）
     换红/蓝底后，1~2px 的半透明边会透出底色，看起来像"蓝色光晕"，
     耳朵、脸颊外缘最明显。原图白底时同样的半透明边看不出来，
     所以这是换底色特有的问题。压窄后实体边缘变实，细发梢渐变仍在。
     用 --no-harden 可关闭对比。

合成阶段用预乘 alpha（render_on_background）：
  先 P=F·α，在预乘空间里裁剪+缩放，再 out = P' + B·(1-α')。
  分开缩放前景和 alpha 会让边缘颜色与覆盖率失配，缩放本身就会造白边。

运行方式：
  python3 main.py camera                 # 摄像头拍照→直接出证件照
  python3 main.py camera --size 2inch --bg blue   # 拍二寸蓝底
  python3 main.py 1inch                  # 一寸，三色全出
  python3 main.py 2inch                  # 二寸
  python3 main.py --bg red               # 只出红底
  python3 main.py --bg 219 142 67        # 自定义 BGR
  python3 main.py --engine classic       # 强制传统 GrabCut
  python3 main.py --debug                # 导出输入/原始掩膜/精修掩膜
  python3 main.py --no-matting           # 关闭 Alpha Matting（做 A/B 对比）

摄像头模式操作：
  预览框就是最终会导出的裁剪范围（绿色=构图可用，黄色=需要靠近/居中）
  空格 拍照 → 回车 生成证件照 → 拍了不满意再按空格重拍 → Q 退出
  原图存到 assets/camera_shot.jpg，成品存到 output/camera_{尺寸}_{底色}.jpg
  预览为 1280×720 采集、按屏幕缩放显示；成品按标准尺寸输出。

依赖说明：
  必需：opencv-python、numpy
  可选：rembg + onnxruntime（AI 抠图，pip install rembg onnxruntime）
        pymatting（Alpha Matting，会自动随 rembg 安装）
  注意：isnet-general-use / silueta 等更强模型需要从 github.com 下载权重，
        本机当前网络访问 github 超时，故默认仍用已缓存的 u2net。

抠图策略演变（踩坑记录，别再走回头路）：
  ❌ GC_INIT_WITH_RECT 铺满整张图 → 背景样本不足，衣服边缘丢
  ❌ GC_INIT_WITH_MASK + 饱和度提示 → 把白底也标成前景，红底变白底
  ❌ 硬编码"非白阈值 248"补衣服 → 只对纯白底成立，换底色就崩
  ❌ 靠膨胀核去"猜"头顶位置补发丝 → 参数对单张图过拟合
  ✅ 传统兜底：GrabCut ∪ 颜色距离 + 最大连通域 + 四角泛洪填洞
  ✅ 主力方案：U2Net 软掩膜 + 颜色去溢出（发丝与边缘光晕同时解决）

TODO（下一迭代）：
  - 交互微调：鼠标拖裁剪框、键盘切底色
  - 自定义背景图：替换纯色为图片
  - 美颜：bilateral 磨皮 + 亮度微调
  - 批量处理 assets/ 目录下全部图片
"""
import os

import cv2
import numpy as np


def imread(path):
    """读取图片，兼容中文路径。

    cv2.imread 在部分平台上处理中文路径会失败，返回 None。
    改用 numpy 读字节流 + cv2.imdecode 解码，最稳。
    """
    data = np.fromfile(path, dtype=np.uint8)
    img = cv2.imdecode(data, cv2.IMREAD_COLOR)
    return img


def imwrite(path, img, quality=95):
    """保存图片，兼容中文路径。

    同样不能用 cv2.imwrite，先编码成内存字节流，再写文件。
    """
    ext = "." + path.rsplit(".", 1)[-1]
    ok, buf = cv2.imencode(ext, img, [int(cv2.IMWRITE_JPEG_QUALITY), quality])
    if not ok:
        raise IOError(f"图片编码失败：{path}")
    buf.tofile(path)


FACE_CASCADE = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"

# 300dpi 下的标准证件照像素尺寸：宽 × 高
SIZE_PRESETS = {
    "1inch": (295, 413),   # 一寸 25×35mm
    "2inch": (413, 579),   # 二寸 35×49mm
}

# 构图参数（可按实际效果微调）
HEAD_TOP_ABOVE_FACE = 0.45   # 发顶在人脸框上沿之上的距离，按脸高比例
HEAD_TOP_MAX_ABOVE_FACE = 1.20  # 掩膜发顶最多高到脸框上方这么远（防止举手入镜）
HEAD_HEIGHT_RATIO = 0.65     # 头部（发顶→下巴）占照片高度的比例
TOP_MARGIN_RATIO = 0.05      # 照片上边缘到发顶的留白比例



def detect_faces(img, min_size=(80, 80)):
    """检测所有人脸，返回 [(x, y, w, h), ...]，左上角坐标 + 宽高。

    先用原灰度图检测；只有检不出时，才用直方图均衡后的灰度图再试一次。
    不能无条件先均衡：实测逆光照片（背后亮窗户/白墙）均衡后整片高光被拉平，
    五官对比度反而消失，Haar 直接检出 0 张——预览能过、生成时却失败。
    均衡只在欠曝场景有用，所以放在失败后兜底。
    """
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    cascade = cv2.CascadeClassifier(FACE_CASCADE)
    if cascade.empty():
        raise RuntimeError(f"级联模型加载失败：{FACE_CASCADE}")

    def _detect(g):
        return cascade.detectMultiScale(
            g,
            scaleFactor=1.1,    # 每次图像缩小 10%，越小越慢但更全
            minNeighbors=5,     # 候选框至少被确认 5 次，越大越严格
            minSize=min_size,   # 过滤过小的误检
        )

    faces = _detect(gray)
    if len(faces) == 0:
        faces = _detect(cv2.equalizeHist(gray))  # 欠曝兜底

    if len(faces) == 0:
        return []
    return [tuple(int(v) for v in f) for f in faces]


def pick_largest_face(faces):
    """证件照场景只保留面积最大的人脸（主体）。"""
    if not faces:
        return None
    return max(faces, key=lambda f: f[2] * f[3])


def draw_face(img, face, color=(0, 255, 0), thickness=3):
    """在原图副本上画出人脸框。"""
    vis = img.copy()
    x, y, w, h = face
    cv2.rectangle(vis, (x, y), (x + w, y + h), color, thickness)
    return vis


def compute_crop_box(face, preset="1inch", person_top=None):
    """由人脸框反推证件照裁剪框，返回 (x, y, w, h)。

    推导过程：
      1. 发顶 = 人脸框上沿 - 0.45×脸高，下巴 = 人脸框下沿
         （传入 person_top 时改用抠图掩膜的真实发顶，见下）
      2. 头部高度固定占照片高度的 65%，反推出照片高度
      3. 按目标宽高比（一寸/二寸都是 5:7）算照片宽度
      4. 水平方向居中于人脸中心，垂直方向让发顶留 5% 白

    person_top：由人像掩膜得到的最上面一行前景 y。
      固定比例在戴耳机、帽子、厚刘海时会严重低估发顶
      （实测戴耳机时偏差 70px，头顶直接被切掉），
      所以有掩膜就以掩膜为准，但限制不能离脸太远（防止举手入镜干扰）。
    """
    x, y, w, h = face
    target_w, target_h = SIZE_PRESETS[preset]
    aspect = target_w / target_h  # 宽高比 ≈ 0.714

    head_top = y - HEAD_TOP_ABOVE_FACE * h
    if person_top is not None:
        head_top = max(person_top, y - HEAD_TOP_MAX_ABOVE_FACE * h)
    chin = y + h
    head_height = chin - head_top

    crop_h = head_height / HEAD_HEIGHT_RATIO
    crop_w = crop_h * aspect

    cx = x + w / 2
    crop_x = cx - crop_w / 2
    crop_y = head_top - TOP_MARGIN_RATIO * crop_h

    return (int(round(crop_x)), int(round(crop_y)),
            int(round(crop_w)), int(round(crop_h)))


def crop_with_padding(img, box, pad_color=(255, 255, 255)):
    """裁剪指定区域；越界时用纯色补边，而不是直接截断。"""
    x, y, w, h = box
    H, W = img.shape[:2]

    left, top = max(0, -x), max(0, -y)
    right, bottom = max(0, x + w - W), max(0, y + h - H)

    if left or top or right or bottom:
        img = cv2.copyMakeBorder(img, top, bottom, left, right,
                                 cv2.BORDER_CONSTANT, value=pad_color)
        x += left
        y += top

    return img[y:y + h, x:x + w]


def draw_box(img, box, color=(0, 0, 255), thickness=3):
    """画出任意矩形框（复用给人脸框/裁剪框）。"""
    vis = img.copy()
    x, y, w, h = box
    cv2.rectangle(vis, (x, y), (x + w, y + h), color, thickness)
    return vis


# GrabCut 初始化参数
# 核心原则：rect 覆盖整个人 + 四周足够背景样本（≥20px）给 GrabCut 建模
GRAB_MIN_MARGIN = 30   # 四周给 GrabCut 留的背景边距（像素），太少学不到背景色
GRAB_X_OUT = 0.45      # 脸框左右各向外扩（总宽=脸宽+左右各45%=约1.9倍脸宽）
GRAB_Y_TOP = 0.55      # 脸框上边向上扩（头顶上方留点）
GRAB_ITER = 5
GRAB_BG_LO = 25        # 与背景色差异低于此值 → 完全透明（背景/阴影）
GRAB_BG_HI = 60        # 与背景色差异高于此值 → 完全不透明（人像）
GRAB_FG_SOLID = 0.85   # 颜色距离 alpha 超过此值判为"确定是人"，用于补 GrabCut 漏掉的衣服

# AI 抠图参数（rembg / U2Net）
AI_MODEL = "u2net"      # u2net / isnet-general-use / silueta
AI_SESSION = None       # 懒加载缓存，加载一次约 2 秒
AI_MIN_ALPHA = 0.25     # 颜色去溢出时 alpha 的下限；太小会把噪声放大成亮点
DECON_BAND = (0.10, 0.95)  # 只在这个 alpha 区间做边缘颜色去溢出
BG_LIKE_THRESH = 18     # 与背景色差异小于此值 → 视为背景像素，强制 alpha=0

# alpha 硬化：把 [LO, HI] 拉成 [0, 1]，压窄过渡带、消除换底色光晕
ALPHA_HARDEN_LO = 0.40
ALPHA_HARDEN_HI = 0.60

# 摄像头请求分辨率（实测本机最高 1280×720）
CAM_W = 1280
CAM_H = 720

# 颜色线投影重解 alpha 的参数
COLOR_ALPHA_SIGMA = 7        # 局部前景色取样半径
COLOR_ALPHA_CONTRAST = 55    # |F-B| 至少这么大，颜色线才可信
COLOR_ALPHA_MIN_CONF = 0.08  # 局部前景色取样置信度下限
COLOR_ALPHA_BAND = 29        # 轮廓带宽度（像素），只在轮廓内外这么大范围内生效


def build_grabcut_rect(face, img_shape):
    """由脸框推算 GrabCut 初始化矩形。

    - 四周留 30px 背景样本给 GrabCut（太窄会学不到背景色）
    - 底部延伸到底图 − margin，确保衣服完整在 rect 内
    - 总宽 = 脸宽 + 左右各 45%，足够覆盖双肩
    """
    x, y, w, h = face
    H, W = img_shape[:2]
    m = GRAB_MIN_MARGIN

    gx = int(x - w * GRAB_X_OUT)
    gy = int(y - h * GRAB_Y_TOP)
    gw = int(w * (1 + 2 * GRAB_X_OUT))
    gh = H - gy - m            # 底部强制到底图边（减 margin 留背景）

    gx = max(m, gx)
    gy = max(m, gy)
    gw = min(gw, W - gx - m)
    gh = max(m, min(gh, H - gy - m))

    return (gx, gy, gw, gh)


def estimate_bg_color(img, band=15, alpha=None):
    """估计原图背景色，返回 BGR float32。

    两种取样方式：
      - 有 alpha 时：只取 alpha < 0.02 的像素（已被判定为背景），最准
      - 无 alpha 时：退化为取四周 band 像素边框的中位数
    取中位数而不是均值，避免肩膀/衣服像素拉偏估计。
    """
    if alpha is not None:
        mask = alpha < 0.02
        if mask.sum() > 500:
            return np.median(img[mask].reshape(-1, 3),
                             axis=0).astype(np.float32)

    H, W = img.shape[:2]
    b = min(band, H // 10, W // 10)
    strips = [
        img[:b, :],          # 上
        img[H - b:, :],      # 下
        img[:, :b],          # 左
        img[:, W - b:],      # 右
    ]
    samples = np.concatenate([s.reshape(-1, 3) for s in strips], axis=0)
    return np.median(samples, axis=0).astype(np.float32)


def segment_person_classic(img, face):
    """传统算法抠图（纯 OpenCV，无需模型）：GrabCut + 颜色距离。

    流程（自适应，不依赖"背景一定是白色"）：
      1. 从图像边缘估计背景色 → 算每个像素"离背景有多远"
      2. GrabCut(GC_INIT_WITH_RECT) 粗分割
      3. 与"远离背景色"的像素取并集 → 补上 GrabCut 漏掉的衣服
      4. 只保留最大连通域，去掉零散误判块
      5. 从图像四角泛洪填洞 → 自动补头发高光/发缝空洞
      6. 用颜色距离做软 alpha → 消除边缘白边，再羽化
    """
    rect = build_grabcut_rect(face, img.shape)
    gx, gy, gw, gh = rect
    H, W = img.shape[:2]

    # ── 1) 背景色 + 颜色距离（越远离背景色越像人）──
    bg_color = estimate_bg_color(img)
    diff = np.abs(img.astype(np.float32) - bg_color).max(axis=2)
    color_alpha = np.clip(
        (diff - GRAB_BG_LO) / (GRAB_BG_HI - GRAB_BG_LO), 0.0, 1.0
    ).astype(np.float32)

    # ── 2) GrabCut 粗分割 ──
    mask = np.zeros((H, W), np.uint8)
    bgd = np.zeros((1, 65), np.float64)
    fgd = np.zeros((1, 65), np.float64)
    cv2.grabCut(img, mask, rect, bgd, fgd, GRAB_ITER, cv2.GC_INIT_WITH_RECT)

    solid = np.where(
        (mask == cv2.GC_FGD) | (mask == cv2.GC_PR_FGD), 255, 0
    ).astype(np.uint8)

    # rect 外强制背景
    solid[:gy, :] = 0
    solid[gy + gh:, :] = 0
    solid[gy:gy + gh, :gx] = 0
    solid[gy:gy + gh, gx + gw:] = 0

    # ── 3) 并集：补 GrabCut 漏掉的衣服（衣服颜色≠背景，颜色距离很大）──
    strong_fg = (color_alpha > GRAB_FG_SOLID).astype(np.uint8) * 255
    strong_fg[:gy, :] = 0
    strong_fg[gy + gh:, :] = 0
    strong_fg[gy:gy + gh, :gx] = 0
    strong_fg[gy:gy + gh, gx + gw:] = 0
    solid = cv2.bitwise_or(solid, strong_fg)

    # ── 4) 只保留最大连通域，去掉零散误判块 ──
    n, labels, stats, _ = cv2.connectedComponentsWithStats(solid, connectivity=8)
    if n > 1:
        biggest = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
        solid = np.where(labels == biggest, 255, 0).astype(np.uint8)

    # ── 5) 闭运算 + 泛洪填洞（补头发高光、发缝等被包住的空洞）──
    solid = cv2.morphologyEx(solid, cv2.MORPH_CLOSE, np.ones((9, 9), np.uint8))

    # 反相后从四角泛洪：能流到的都是真背景，流不到的 255 就是被包住的洞
    holes = cv2.bitwise_not(solid)
    ff_mask = np.zeros((H + 2, W + 2), np.uint8)
    for seed in [(0, 0), (W - 1, 0), (0, H - 1), (W - 1, H - 1)]:
        cv2.floodFill(holes, ff_mask, seed, 0)
    solid = cv2.bitwise_or(solid, holes)

    # ── 6) 软 alpha：solid 外扩 1px 内按颜色距离取不透明度 ──
    outer = cv2.dilate(solid, np.ones((3, 3), np.uint8), iterations=1) > 0
    alpha = np.where(outer, color_alpha, 0.0).astype(np.float32)

    # 内部核心区强制全不透明（头发/皮肤高光虽然是浅色但确实属于人）
    core = cv2.erode(solid, np.ones((5, 5), np.uint8), iterations=2)
    alpha[core > 0] = 1.0

    # ── 7) 轻微羽化，让边缘过渡自然 ──
    alpha = cv2.GaussianBlur(alpha, (3, 3), 0)
    return np.clip(alpha, 0.0, 1.0)


def ai_engine_available():
    """检测 rembg + onnxruntime 是否可用（不加载模型，只查包）。"""
    try:
        import importlib.util
        return (
            importlib.util.find_spec("rembg") is not None
            and importlib.util.find_spec("onnxruntime") is not None
        )
    except Exception:
        return False


def get_ai_session():
    """懒加载 rembg 会话（首次调用约 2 秒，且首次会下载 ~176MB 模型）。"""
    global AI_SESSION
    if AI_SESSION is None:
        from rembg import new_session
        print(f"正在加载 AI 抠图模型（{AI_MODEL}）……")
        AI_SESSION = new_session(AI_MODEL)
    return AI_SESSION


def segment_person_ai(img):
    """AI 抠图（rembg / U2Net），返回 float32 alpha mask，值范围 [0, 1]。

    相比 GrabCut 的根本优势：U2Net 是端到端训练的显著性分割网络，
    能识别"细到几个像素的发丝"，不依赖颜色分布假设。
    输出本身就是软 alpha（0~1 连续值），不做二值化，保留半透明发丝。
    """
    from rembg import remove

    rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    rgba = remove(rgb, session=get_ai_session())
    alpha = rgba[:, :, 3].astype(np.float32) / 255.0
    return np.clip(alpha, 0.0, 1.0)


def matting_available():
    """检测 pymatting 是否可用（rembg 的依赖，通常已随 rembg 装上）。"""
    import importlib.util
    return importlib.util.find_spec("pymatting") is not None


def refine_alpha_matting(img, alpha):
    """Alpha Matting 精修：把 U2Net 的模糊边界解算成真实覆盖率。

    为什么需要这一步：
      U2Net 输出是 320×320 上采样回原尺寸的，边界被抹成 5~7px 宽的斜坡。
      实测在纯白背景像素上会给出 α=0.18~0.65 的错误高估值，
      这些像素贴到红底上就变成一圈白雾。

    做法（closed-form matting，Levin 2007）：
      用原 alpha 建 trimap —— >0.9 确定前景、<0.1 确定背景、中间是未知带；
      然后在未知带内用图像自身颜色解 α，把"纯属背景色"的像素压到 0。

    需要 pymatting；不可用时原样返回（不影响主流程）。
    """
    if not matting_available():
        return alpha

    from pymatting import estimate_alpha_cf

    trimap = np.full(alpha.shape, 0.5, dtype=np.float64)
    trimap[alpha > 0.90] = 1.0
    trimap[alpha < 0.10] = 0.0

    rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float64) / 255.0
    refined = estimate_alpha_cf(rgb, trimap)
    return np.clip(refined.astype(np.float32), 0.0, 1.0)


def suppress_background_like(img, alpha, bg_color):
    """背景色先验：颜色几乎等于原背景色的像素强制透明。

    阈值 0.99 而不是 0.95：U2Net 会给"发丝间隙透出的纯白背景"打上 α=0.92。

    只按颜色不判连通性——曾经试过"只有与图像边框连通的才清零"，
    结果头发丝之间的背景白被当成"高光"保留下来，头顶出现一块白斑。
    """
    diff = np.abs(img.astype(np.float32) - bg_color).max(axis=2)
    hit = (alpha < 0.99) & (diff < BG_LIKE_THRESH)
    out = alpha.copy()
    out[hit] = 0.0
    return out


def harden_alpha(alpha):
    """压窄 alpha 过渡带，让实体边缘变实，消除换底色后的半透明光晕。

    为什么需要：
      背景一旦换成饱和的红/蓝，1~2px 的半透明带会透出底色，
      看起来就是耳朵、脸颊外缘一圈"蓝色光晕"。原图是白底，
      同样的半透明带因为和肤色接近，根本看不出来。
      U2Net 的 α 来自 320×320 上采样，边缘天然是 5~6px 斜坡，
      在 295×413 的输出上仍有 2~3px，正好是可见的光晕宽度。

    做法：把 α∈[0.40, 0.60] 线性拉伸到 [0, 1]，其余钳位。
      过渡带宽度砍到约 1/5，实体边缘变硬；
      α<0.4 的细发梢仍保留渐变，不会被一刀切掉。
    """
    lo, hi = ALPHA_HARDEN_LO, ALPHA_HARDEN_HI
    return np.clip((alpha - lo) / (hi - lo), 0.0, 1.0).astype(np.float32)


def refine_alpha_by_color(img, alpha, bg_color):
    """在轮廓带内用颜色线投影重解 alpha —— 本次修复白边的核心。

    原理：合成方程 I = αF + (1-α)B 里，若已知背景色 B 和局部前景色 F，
    α 就由像素颜色唯一确定：
        α = <I-B, F-B> / |F-B|²
    这是把 I 投影到 (B → F) 这条颜色线上得到的覆盖率。

    为什么必须有这一步：
      U2Net 输出是 320×320 上采样，在真实发际线上系统性高估 α。
      实测纯白背景像素 α=0.18~0.65，发丝间隙的纯白像素甚至给到 0.92。
      这些像素做颜色去溢出会算出 F≈255 的假前景，贴到红/蓝底上就是白雾。
      模型 alpha 只能"猜形状"，颜色线投影能"算覆盖率"。

    作用范围：只在轮廓内外的窄带生效（长头发内部的高光不会被误判），
    且要求前后景颜色对比足够大，否则颜色线不可靠、保留模型 alpha。
    """
    I = img.astype(np.float32)

    # 局部前景色 F：对确定前景像素做加权高斯平均
    conf = (alpha > 0.95).astype(np.float32)
    w = cv2.erode(conf, np.ones((5, 5), np.uint8))
    den = cv2.GaussianBlur(w, (0, 0), COLOR_ALPHA_SIGMA)
    f_local = cv2.GaussianBlur(I * w[..., None], (0, 0), COLOR_ALPHA_SIGMA)
    f_local /= np.maximum(den[..., None], 1e-4)

    d = f_local - bg_color
    dd = (d * d).sum(axis=2)
    a_col = np.clip(((I - bg_color) * d).sum(axis=2) / np.maximum(dd, 1e-4), 0.0, 1.0)

    # 只在轮廓带生效：深内部（长发高光）与远景（纯背景）都排除
    solid = (alpha > 0.5).astype(np.uint8)
    deep = cv2.erode(solid, np.ones((COLOR_ALPHA_BAND, COLOR_ALPHA_BAND),
                                   np.uint8)) > 0
    near = cv2.dilate(solid, np.ones((COLOR_ALPHA_BAND, COLOR_ALPHA_BAND),
                                     np.uint8)) > 0
    use = (near & ~deep
           & (dd > COLOR_ALPHA_CONTRAST ** 2)
           & (den > COLOR_ALPHA_MIN_CONF))

    out = alpha.copy()
    out[use] = np.minimum(alpha[use], a_col[use])
    return out



def decontaminate_colors(img, alpha, bg_color):
    """边缘颜色去溢出：把半透明像素里混入的原背景色剥掉。

    原理（unpremultiply）：原来的观测像素
        I = α·F + (1-α)·B
    已知背景色 B（原图边缘估计得到），反解真实前景色
        F = (I - (1-α)·B) / α
    这样贴到红/蓝底上时，发丝边缘不会带一圈原背景的白色光晕。
    """
    lo, hi = DECON_BAND
    band = (alpha > lo) & (alpha < hi)
    if not np.any(band):
        return img.astype(np.float32)

    a = np.clip(alpha, AI_MIN_ALPHA, 1.0)[..., None]
    fg_est = (img.astype(np.float32) - (1.0 - a) * bg_color) / a
    fg_est = np.clip(fg_est, 0, 255)

    out = img.astype(np.float32)
    out[band] = fg_est[band]
    return out


def render_on_background(fg_img, alpha, box, size, bg_color):
    """按预乘 alpha（premultiplied alpha）裁剪 → 缩放 → 合成为纯色背景。

    为什么不能"先各自缩放再合成"：
      前景色和 alpha 分开做 INTER_AREA 时，边缘像素的颜色会被旁边的
      背景色稀释，而 alpha 不会同步变化，两者不再满足 I = αF + (1-α)B，
      缩放本身就会在边缘留下一圈浅色。正确做法是在预乘空间里缩放：
          P = F·α   →   P' = resize(P)     α' = resize(α)
          out = P' + B·(1-α')
      这样缩放与合成都是线性的，边缘颜色与覆盖率保持一致。
    """
    fg_crop = crop_with_padding(fg_img, box)
    a_crop = crop_with_padding(alpha[..., None], box, pad_color=0.0)[..., 0]

    premul = fg_crop * a_crop[..., None]
    premul_small = cv2.resize(premul, size, interpolation=cv2.INTER_AREA)
    alpha_small = cv2.resize(a_crop, size, interpolation=cv2.INTER_AREA)

    bg = np.asarray(bg_color, dtype=np.float32)
    out = premul_small + bg * (1.0 - alpha_small[..., None])
    return np.clip(out, 0, 255).astype(np.uint8), alpha_small


def segment_person(img, face, engine="auto"):
    """抠图统一入口，返回 float32 alpha mask [0, 1]。

    engine:
      auto    → 有 rembg 就用 AI，没有就退回传统算法
      ai      → 强制 AI（缺依赖时报错）
      classic → 强制传统算法（纯 OpenCV，便于离线/对比）
    """
    if engine == "auto":
        engine = "ai" if ai_engine_available() else "classic"

    if engine == "ai":
        if not ai_engine_available():
            raise RuntimeError(
                "AI 抠图需要 rembg 与 onnxruntime：pip install rembg onnxruntime"
            )
        return segment_person_ai(img)

    return segment_person_classic(img, face)


def composite_on_bg(fg_img, alpha, bg_color=(255, 255, 255)):
    """按 alpha 把前景贴到纯色背景上，返回合成后的 BGR 图像。

    alpha 是单通道 float32，值域 [0, 1]；bg_color 是 BGR 三元组。
    """
    alpha3 = alpha[..., None]  # (H,W) → (H,W,1) 方便广播
    bg = np.full_like(fg_img, bg_color, dtype=np.uint8)
    out = fg_img * alpha3 + bg * (1 - alpha3)
    return out.astype(np.uint8)


def composite_on_image(fg_img, alpha, bg_img):
    """按 alpha 把前景贴到背景图上（需保证同尺寸）。"""
    alpha3 = alpha[..., None]
    out = fg_img * alpha3 + bg_img.astype(np.float32) * (1 - alpha3)
    return out.astype(np.uint8)

def show_image(img, title="Preview"):
    """显示图片，在**图片窗口**内按任意键或点右上角 X 关闭。

    注意：cv2.waitKey 监听的是 OpenCV 窗口的按键，不是终端 stdin。
    所以窗口弹出期间在终端里敲 q / exit / Ctrl+C 都不会被接收，
    必须在图片窗口上按键，或直接点窗口的关闭按钮。
    这里轮询窗口可见性，点 X 也能立刻退出，不会死等。
    """
    win = f"{title}  —— 按任意键 / 点 X 关闭"
    cv2.imshow(win, img)
    print(f"预览窗口「{win}」已弹出：请切到该窗口按任意键关闭（在终端里按 q/Ctrl+C 无效）")
    while True:
        if cv2.waitKey(30) != -1:
            break
        if cv2.getWindowProperty(win, cv2.WND_PROP_VISIBLE) < 1:
            break
    cv2.destroyAllWindows()


BG_LABELS = {"white": "白底", "red": "红底", "blue": "蓝底"}


def make_contact_sheet(items, gap=12):
    """把多种底色的成品横向拼成一张，一次看完，而不是只看最后一张。"""
    tiles = []
    for name, photo in items:
        label = BG_LABELS.get(name, name)
        tile = cv2.copyMakeBorder(photo, 36, 0, 0, 0, cv2.BORDER_CONSTANT,
                                  value=(240, 240, 240))
        tile = put_text_cn(tile, label, (10, 5), 24, (40, 40, 40))
        tiles.append(tile)

    tallest = max(t.shape[0] for t in tiles)
    tiles = [cv2.copyMakeBorder(t, 0, tallest - t.shape[0], 0, 0,
                                cv2.BORDER_CONSTANT, value=(240, 240, 240))
             for t in tiles]

    sep = np.full((tallest, gap, 3), 60, np.uint8)
    sheet = tiles[0]
    for tile in tiles[1:]:
        sheet = np.hstack([sheet, sep, tile])
    return sheet


CN_FONT_CANDIDATES = [
    "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
    "/usr/share/fonts/opentype/noto/NotoSansCJK-Medium.ttc",
    "/usr/share/fonts/truetype/arphic/uming.ttc",
]
CN_FONT_PATH = next((p for p in CN_FONT_CANDIDATES if os.path.exists(p)), None)
_CN_FONT_CACHE = {}


def _load_cn_font(size):
    """按字号缓存字体对象——每帧重新加载要 10ms 以上，预览会卡。"""
    if size not in _CN_FONT_CACHE:
        from PIL import ImageFont
        _CN_FONT_CACHE[size] = ImageFont.truetype(CN_FONT_PATH, size)
    return _CN_FONT_CACHE[size]


def put_text_cn(img, text, org, size=22, color=(255, 255, 255)):
    """在图像上写中文（cv2.putText 不支持 CJK，会显示成 ?）。

    用 PIL 渲染；系统没有中文字体时退回英文提示，不会崩。
    """
    if CN_FONT_PATH is None:
        cv2.putText(img, "SPACE: shoot   ENTER: make photo   Q: quit",
                    (org[0], org[1] + size), cv2.FONT_HERSHEY_SIMPLEX,
                    0.5, color, 1, cv2.LINE_AA)
        return img

    from PIL import Image, ImageDraw
    pil = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(pil)
    font = _load_cn_font(size)
    x, y = org
    draw.text((x + 1, y + 1), text, font=font, fill=(0, 0, 0))   # 描边，深色背景也看得清
    draw.text((x, y), text, font=font, fill=color[::-1])          # BGR → RGB
    return cv2.cvtColor(np.array(pil), cv2.COLOR_RGB2BGR)


def draw_camera_guide(frame, box, face, hit):
    """预览叠加层：画出最终会导出的裁剪范围 + 人脸框 + 状态提示。"""
    disp = frame.copy()
    color = (0, 255, 0) if hit else (0, 220, 255)

    if box is not None:
        bx, by, bw, bh = box
        cv2.rectangle(disp, (bx, by), (bx + bw, by + bh), color, 2)
        # 头顶参考线和眼部参考线，方便对齐
        cv2.line(disp, (bx, by + int(bh * 0.08)), (bx + bw, by + int(bh * 0.08)),
                 color, 1)
        cv2.line(disp, (bx, by + int(bh * 0.42)), (bx + bw, by + int(bh * 0.42)),
                 color, 1)

    if face is not None:
        x, y, fw, fh = face
        cv2.rectangle(disp, (x, y), (x + fw, y + fh), (0, 255, 0), 1)

    msg = "位置合适，按空格拍照" if hit else "请靠近并正对镜头，让头肩进入方框"
    return put_text_cn(disp, msg, (20, 16), 24, color)


def draw_tips(frame, text):
    """底部半透明提示条。"""
    h, w = frame.shape[:2]
    bar = frame.copy()
    cv2.rectangle(bar, (0, h - 46), (w, h), (0, 0, 0), -1)
    disp = cv2.addWeighted(bar, 0.55, frame, 0.45, 0)
    return put_text_cn(disp, text, (16, h - 36), 22, (255, 255, 255))


def guide_hit(box, face, frame_shape):
    """判断当前构图是否可用：人脸够大，且裁剪框完整落在画面内。"""
    if face is None or box is None:
        return False
    h, w = frame_shape[:2]
    fw = face[2]
    if not (0.14 * w <= fw <= 0.45 * w):
        return False
    bx, by, bw, bh = box
    return bx >= 0 and by >= 0 and bx + bw <= w and by + bh <= h


def scale_for_screen(frame, max_h=720):
    """预览缩放到屏幕能放下的尺寸（保存的仍是原始分辨率）。"""
    h, w = frame.shape[:2]
    if h <= max_h:
        return frame, 1.0
    s = max_h / h
    return cv2.resize(frame, (int(w * s), int(h * s)), interpolation=cv2.INTER_AREA), s


def capture_from_camera(index=0, save_path="assets/camera_shot.jpg",
                        mirror=True, preset="1inch"):
    """打开摄像头预览：空格拍照，回车用拍到的照片生成证件照，Q/Esc 退出。

    预览里画的方框就是最终会导出的裁剪范围，所见即所得。
    mirror=True 做水平翻转，和手机自拍预览一致。
    返回拍到的 BGR 图像（已镜像、无辅助线）；退出则返回 None。
    """
    cap = cv2.VideoCapture(index, cv2.CAP_V4L2)
    if not cap.isOpened():
        cap = cv2.VideoCapture(index)  # 退回到默认后端
    if not cap.isOpened():
        print(f"错误：无法打开摄像头 {index}，检查设备是否被占用")
        return None

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, CAM_W)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAM_H)
    real_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    real_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(f"摄像头已打开：{real_w}×{real_h}")

    shot = None
    face = None
    box = None
    tick = 0
    print("空格拍照 / 回车生成证件照 / 拍了不满意再按空格重拍 / Q 退出")

    while True:
        ok, frame = cap.read()
        if not ok:
            print("错误：读取摄像头画面失败")
            break
        if mirror:
            frame = cv2.flip(frame, 1)

        if shot is None:
            tick += 1
            if tick % 4 == 1:  # 每 4 帧检测一次，够跟手也不卡
                face = pick_largest_face(detect_faces(frame))
                box = compute_crop_box(face, preset) if face is not None else None
            hit = guide_hit(box, face, frame.shape)
            disp = draw_camera_guide(frame, box, face, hit)
            disp = draw_tips(disp, "空格 拍照      Q 退出")
        else:
            disp = draw_tips(shot, "回车 生成证件照      空格 重拍      Q 退出")

        view, _ = scale_for_screen(disp, 700)
        cv2.imshow("ID Photo Camera", view)

        key = cv2.waitKey(1) & 0xFF
        if key in (ord("q"), 27):
            shot = None
            break
        if key == 32:  # 空格
            if shot is None:
                if face is None:
                    print("没检测到人脸，请正对镜头后再拍")
                    continue
                if not hit:
                    print("构图不理想（人脸太小或超出画面），已拍下，可按空格重拍")
                shot = frame.copy()
                imwrite(save_path, shot)
                print(f"已拍照并保存原图：{save_path}")
            else:
                shot = None
                print("重拍中……")
        elif key in (10, 13) and shot is not None:  # 回车
            break

    cap.release()
    cv2.destroyAllWindows()
    return shot



def process_photo(img, preset, bg_list, engine="auto", debug=False,
                  matting=True, harden=True, out_prefix=None):
    """把一张照片处理成证件照，返回最后一张成品（BGR）。

    摄像头拍摄和本地图片走同一套流程，避免两份实现走偏。
    """
    target_w, target_h = SIZE_PRESETS[preset]
    prefix = out_prefix or preset
    img_h, img_w = img.shape[:2]
    print(f"输入尺寸：宽 {img_w}，高 {img_h}")

    faces = detect_faces(img)
    print(f"检测到 {len(faces)} 张人脸")
    face = pick_largest_face(faces)
    if face is None:
        print("没有检测到人脸，无法生成证件照")
        return None
    print(f"主脸位置：x={face[0]}, y={face[1]}, 宽={face[2]}, 高={face[3]}")

    # ── 抠图（先抠图，才能知道真实发顶在哪）──
    used_engine = engine
    if used_engine == "auto":
        used_engine = "ai" if ai_engine_available() else "classic"
    print(f"正在抠图（{'AI / U2Net' if used_engine == 'ai' else '传统 GrabCut'}）……")
    alpha_raw = segment_person(img, face, engine=used_engine)
    print(f"原始前景占比: {alpha_raw.mean()*100:.1f}%")

    # ── 用掩膜找真实发顶，修正裁剪框 ──
    fx, fy, fw, fh = face
    sx0 = max(0, int(fx - 0.35 * fw))
    sx1 = min(img_w, int(fx + 1.35 * fw))
    rows = np.where((alpha_raw[:, sx0:sx1] > 0.5).any(axis=1))[0]
    person_top = float(rows.min()) if rows.size else None

    # ── 构图预览 ──
    box = compute_crop_box(face, preset, person_top=person_top)
    vis = draw_box(draw_face(img, face), box, color=(0, 0, 255))
    imwrite("output/composition_preview.jpg", vis)
    print(f"裁剪框（{preset}）：{box}  发顶 y={person_top}")

    # 背景色用 alpha 判定过的真背景像素取样，比边框取样准
    bg_color = estimate_bg_color(img, alpha=alpha_raw)

    # ── 精修 1：Alpha Matting，把 320×320 上采样的模糊边界解成真实覆盖率 ──
    alpha = refine_alpha_matting(img, alpha_raw) if matting else alpha_raw

    # ── 精修 2：颜色线投影重解过渡带 alpha（修白边的核心步骤）──
    alpha = refine_alpha_by_color(img, alpha, bg_color)

    # ── 精修 3：背景色先验，颜色等于原背景的像素强制透明 ──
    alpha = suppress_background_like(img, alpha, bg_color)

    # ── 精修 4：压窄过渡带，消除换底色后边缘透出的光晕 ──
    if harden:
        alpha = harden_alpha(alpha)

    # ── 精修 5：边缘颜色去溢出，剥掉半透明像素里混入的原背景色 ──
    fg_clean = decontaminate_colors(img, alpha, bg_color)
    print(f"精修后前景占比: {(alpha > 0.5).mean()*100:.1f}%")

    # ── 调试：导出可直接对照的中间结果 ──
    if debug:
        os.makedirs("output/debug", exist_ok=True)
        imwrite("output/debug/01_input.png", img)
        imwrite("output/debug/02_mask_raw.png", (alpha_raw * 255).astype(np.uint8))
        imwrite("output/debug/03_mask_refined.png",
                (alpha * 255).astype(np.uint8))
        print("  ✓ output/debug/01_input.png（原始输入）")
        print("  ✓ output/debug/02_mask_raw.png（原始分割掩膜）")
        print("  ✓ output/debug/03_mask_refined.png（精修后掩膜）")

    # ── 合成：预乘 alpha 空间内裁剪 + 缩放，避免缩放本身产生白边 ──
    names = ", ".join(n for n, _ in bg_list)
    print(f"正在合成底色（{names}）……")
    photos = []
    for name, bgr in bg_list:
        photo, _ = render_on_background(
            fg_clean, alpha, box, (target_w, target_h), bgr
        )
        out_path = f"output/{prefix}_{name}.jpg"
        imwrite(out_path, photo)
        print(f"  ✓ {out_path}  尺寸 {target_w}×{target_h}  BGR{bgr}")
        photos.append((name, photo))

    if not photos:
        return None

    # 先打印文件清单，再弹预览窗口（窗口会阻塞，先打印才看得到）
    print(f"共生成 {len(photos)} 张：")
    for name, _ in photos:
        print(f"  output/{prefix}_{name}.jpg")

    # 一次显示全部底色，而不是只显示最后一张
    if len(photos) == 1:
        show_image(photos[0][1], f"{prefix} - {photos[0][0]}")
    else:
        show_image(make_contact_sheet(photos), f"{prefix} - 全部 {len(photos)} 种底色")

    return photos[-1][1]


def main():
    # ── 命令行参数解析 ──
    import argparse
    parser = argparse.ArgumentParser(description="证件照自动裁剪与背景替换工具")
    parser.add_argument("preset", nargs="?", default="1inch",
                        choices=["camera", "1inch", "2inch"],
                        help="preset: camera / 1inch / 2inch（默认 1inch）")
    parser.add_argument("--bg", nargs="+", default=None,
                        help="背景色：预设名（white/red/blue）或任意 3 个 BGR 整数，如 --bg 219 142 67；不传则三预设全出")
    parser.add_argument("--engine", default="auto", choices=["auto", "ai", "classic"],
                        help="抠图引擎：auto（默认，有 rembg 用 AI）/ ai / classic（纯 OpenCV）")
    parser.add_argument("--debug", action="store_true",
                        help="导出中间结果到 output/debug/（输入、原始掩膜、精修掩膜）")
    parser.add_argument("--no-matting", dest="matting", action="store_false",
                        help="关闭 Alpha Matting 精修（可用于 A/B 对比）")
    parser.add_argument("--size", default="1inch", choices=["1inch", "2inch"],
                        help="camera 模式下出几寸（默认 1inch）")
    parser.add_argument("--no-harden", dest="harden", action="store_false",
                        help="关闭 alpha 硬化（观察边缘光晕对比用）")
    args = parser.parse_args()

    # 背景色预设（BGR）
    BG_PRESETS = {
        "white": (255, 255, 255),
        "red":   (66, 66, 244),   # 证件红 BGR
        "blue":  (219, 142, 67),  # 常用证件蓝 BGR
    }

    # 解析 --bg 参数 → 要合成的 (name, bgr) 列表
    if args.bg is None:
        # 不传 → 三预设全出
        bg_list = list(BG_PRESETS.items())
    elif len(args.bg) == 1 and args.bg[0].lower() in BG_PRESETS:
        # 单个预设名
        name = args.bg[0].lower()
        bg_list = [(name, BG_PRESETS[name])]
    elif len(args.bg) == 3:
        # 三个整数 → 自定义 BGR
        try:
            bgr = tuple(max(0, min(255, int(x))) for x in args.bg)
            bg_list = [(f"custom_{'_'.join(str(v) for v in bgr)}", bgr)]
        except ValueError:
            parser.error("--bg 后跟 3 个 0-255 整数（BGR），或预设名 white/red/blue")
    else:
        parser.error("--bg 参数格式：预设名（white/red/blue）或 3 个 BGR 整数")

    # ── 摄像头：拍完直接出证件照 ──
    if args.preset == "camera":
        shot = capture_from_camera(preset=args.size)
        if shot is None:
            print("已退出，未生成证件照")
            return
        process_photo(shot, args.size, bg_list,
                      engine=args.engine, debug=args.debug,
                      matting=args.matting, harden=args.harden,
                      out_prefix=f"camera_{args.size}")
        return

    preset = args.preset

    # ── 读图 ──
    path = "assets/1.png"
    img = imread(path)
    if img is None:
        print(f"读取失败：{path}，请检查路径或文件是否损坏")
        return

    process_photo(img, preset, bg_list,
                  engine=args.engine, debug=args.debug,
                  matting=args.matting, harden=args.harden)


if __name__ == "__main__":
    main()
