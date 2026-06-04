import { useEffect, useRef, useState } from 'react'
import {
  FilesetResolver,
  ObjectDetector,
  GestureRecognizer,
  ImageSegmenter,
  type ObjectDetectorResult,
  type GestureRecognizerResult,
  type ImageSegmenterResult,
} from '@mediapipe/tasks-vision'

const WASM_URL = 'https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.35/wasm'
const OBJ_MODEL = 'https://storage.googleapis.com/mediapipe-models/object_detector/efficientdet_lite0/float32/1/efficientdet_lite0.tflite'
const GESTURE_MODEL = 'https://storage.googleapis.com/mediapipe-models/gesture_recognizer/gesture_recognizer/float16/1/gesture_recognizer.task'
const SEG_MODEL = 'https://storage.googleapis.com/mediapipe-models/image_segmenter/selfie_segmenter/float16/1/selfie_segmenter.tflite'

const MAX_DETECTIONS = 20
const TRACK_TIMEOUT_MS = 800
const TRACK_MATCH_IOU = 0.2
const SCORE_THRESHOLD = 0.45
const POSITION_ALPHA = 0.35
const SCORE_ALPHA = 0.15
const ENTRY_ANIM_MS = 600         // 트랙 첫 등장 시 페이드인 + 스케일업
const PALM_PROXIMITY_RATIO = 0.6  // 손 → 트랙 bbox 중심 거리 임계 (bbox 너비 대비)
const STILLNESS_DWELL_MS = 1500   // 정지 판단 임계
const STILLNESS_MOVE_PX = 25      // 이만큼 안 움직이면 정지

type Status = 'idle' | 'loading-model' | 'requesting-camera' | 'running' | 'error'
type BBox = { x: number; y: number; w: number; h: number }
type EffectType = 'halo' | 'sequence' | 'ring' | 'particles'
type ShapeMode = 'box' | 'silhouette'
type InteractionMode = 'none' | 'palm-hide' | 'stillness-boost' | 'jump-grow'

type Track = {
  id: number
  bbox: BBox
  score: number
  firstSeenAt: number
  lastSeenAt: number
  effectVisibility: number    // 0..1 — 상호작용 모드에서 사용
  lastCenter: { x: number; y: number; ts: number }
  stillSince: number | null
  prevTopY: number | null
  jumpBoost: number           // 0..1
}

export default function App() {
  const videoRef = useRef<HTMLVideoElement>(null)
  const canvasRef = useRef<HTMLCanvasElement>(null)
  const detectorRef = useRef<ObjectDetector | null>(null)
  const gestureRef = useRef<GestureRecognizer | null>(null)
  const segmenterRef = useRef<ImageSegmenter | null>(null)

  const [status, setStatus] = useState<Status>('idle')
  const [errorMsg, setErrorMsg] = useState('')
  const [mirror, setMirror] = useState(true)
  const [showOverlay, setShowOverlay] = useState(true)
  const [debug, setDebug] = useState(false)
  const [fps, setFps] = useState(0)
  const [trackCount, setTrackCount] = useState(0)
  const [effect, setEffect] = useState<EffectType>('halo')
  const [shape, setShape] = useState<ShapeMode>('box')
  const [interaction, setInteraction] = useState<InteractionMode>('none')

  const refs = {
    mirror: useRef(mirror),
    showOverlay: useRef(showOverlay),
    debug: useRef(debug),
    effect: useRef(effect),
    shape: useRef(shape),
    interaction: useRef(interaction),
  }
  useEffect(() => { refs.mirror.current = mirror }, [mirror])
  useEffect(() => { refs.showOverlay.current = showOverlay }, [showOverlay])
  useEffect(() => { refs.debug.current = debug }, [debug])
  useEffect(() => { refs.effect.current = effect }, [effect])
  useEffect(() => { refs.shape.current = shape }, [shape])
  useEffect(() => { refs.interaction.current = interaction }, [interaction])

  const tracksRef = useRef<Track[]>([])
  const nextIdRef = useRef(1)
  const lastGestureRef = useRef<GestureRecognizerResult | null>(null)
  const lastSegMaskRef = useRef<ImageSegmenterResult | null>(null)

  // 단축키 (IME 독립 — e.code 사용)
  useEffect(() => {
    function onKey(e: KeyboardEvent) {
      if (e.target instanceof HTMLInputElement || e.target instanceof HTMLTextAreaElement) return
      switch (e.code) {
        case 'KeyD': setDebug((v) => !v); break
        case 'KeyM': setMirror((v) => !v); break
        case 'KeyO': setShowOverlay((v) => !v); break
        case 'KeyF': toggleFullscreen(); break
        case 'KeyB': setShape('box'); break
        case 'KeyS': setShape('silhouette'); break
        case 'Digit1': setEffect('halo'); break
        case 'Digit2': setEffect('sequence'); break
        case 'Digit3': setEffect('ring'); break
        case 'Digit4': setEffect('particles'); break
        case 'KeyN': setInteraction('none'); break
        case 'KeyP': setInteraction('palm-hide'); break
        case 'KeyT': setInteraction('stillness-boost'); break
        case 'KeyJ': setInteraction('jump-grow'); break
      }
    }
    window.addEventListener('keydown', onKey)
    return () => window.removeEventListener('keydown', onKey)
  }, [])

  useEffect(() => {
    let cancelled = false
    let raf: number | null = null
    let lastTs = performance.now()
    const fpsBuf: number[] = []

    async function init() {
      try {
        setStatus('loading-model')
        const vision = await FilesetResolver.forVisionTasks(WASM_URL)

        const [detector, gesture, segmenter] = await Promise.all([
          ObjectDetector.createFromOptions(vision, {
            baseOptions: { modelAssetPath: OBJ_MODEL, delegate: 'GPU' },
            runningMode: 'VIDEO',
            scoreThreshold: SCORE_THRESHOLD,
            maxResults: MAX_DETECTIONS,
            categoryAllowlist: ['person'],
          }),
          GestureRecognizer.createFromOptions(vision, {
            baseOptions: { modelAssetPath: GESTURE_MODEL, delegate: 'GPU' },
            runningMode: 'VIDEO',
            numHands: 4,
          }),
          ImageSegmenter.createFromOptions(vision, {
            baseOptions: { modelAssetPath: SEG_MODEL, delegate: 'GPU' },
            runningMode: 'VIDEO',
            outputCategoryMask: true,
            outputConfidenceMasks: false,
          }),
        ])

        if (cancelled) {
          detector.close(); gesture.close(); segmenter.close()
          return
        }
        detectorRef.current = detector
        gestureRef.current = gesture
        segmenterRef.current = segmenter

        setStatus('requesting-camera')
        const stream = await navigator.mediaDevices.getUserMedia({
          video: { width: { ideal: 1280 }, height: { ideal: 720 }, facingMode: 'user' },
          audio: false,
        })
        if (cancelled) {
          stream.getTracks().forEach((t) => t.stop())
          return
        }

        const video = videoRef.current!
        video.srcObject = stream
        await video.play()
        setStatus('running')

        const loop = () => {
          if (cancelled) return
          const now = performance.now()
          const dt = now - lastTs
          lastTs = now
          if (dt > 0) {
            fpsBuf.push(1000 / dt)
            if (fpsBuf.length > 30) fpsBuf.shift()
          }
          if (fpsBuf.length === 30) {
            const avg = fpsBuf.reduce((a, b) => a + b, 0) / fpsBuf.length
            setFps(Math.round(avg))
          }
          detectAndRender(now)
          raf = requestAnimationFrame(loop)
        }
        raf = requestAnimationFrame(loop)
      } catch (e) {
        if (cancelled) return
        setStatus('error')
        setErrorMsg(e instanceof Error ? e.message : String(e))
      }
    }
    init()

    return () => {
      cancelled = true
      if (raf) cancelAnimationFrame(raf)
      const video = videoRef.current
      if (video?.srcObject) {
        ;(video.srcObject as MediaStream).getTracks().forEach((t) => t.stop())
        video.srcObject = null
      }
      detectorRef.current?.close(); detectorRef.current = null
      gestureRef.current?.close(); gestureRef.current = null
      segmenterRef.current?.close(); segmenterRef.current = null
    }
  }, [])

  function detectAndRender(ts: number) {
    const video = videoRef.current
    const canvas = canvasRef.current
    const detector = detectorRef.current
    const gesture = gestureRef.current
    const segmenter = segmenterRef.current
    if (!video || !canvas || !detector || !gesture || !segmenter) return
    if (video.readyState < 2) return

    const vw = video.videoWidth
    const vh = video.videoHeight
    if (canvas.width !== vw) canvas.width = vw
    if (canvas.height !== vh) canvas.height = vh

    const ctx = canvas.getContext('2d')!
    const mirrored = refs.mirror.current

    // 거울 적용 후 카메라 그림
    ctx.save()
    if (mirrored) { ctx.translate(vw, 0); ctx.scale(-1, 1) }
    ctx.drawImage(video, 0, 0, vw, vh)
    ctx.restore()

    // ─── Object detection (다인) ───
    let objResult: ObjectDetectorResult | undefined
    try { objResult = detector.detectForVideo(video, ts) } catch { return }
    const detections: { bbox: BBox; score: number }[] = []
    if (objResult) {
      for (const d of objResult.detections) {
        const c = d.categories?.[0]
        if (!c || c.categoryName !== 'person' || c.score < SCORE_THRESHOLD) continue
        const b = d.boundingBox
        if (!b) continue
        detections.push({ bbox: { x: b.originX, y: b.originY, w: b.width, h: b.height }, score: c.score })
      }
    }

    updateTracks(tracksRef.current, detections, ts, nextIdRef)
    if (tracksRef.current.length !== trackCount) setTrackCount(tracksRef.current.length)

    // ─── Gesture detection (상호작용용) ───
    let gestureResult: GestureRecognizerResult | undefined
    try { gestureResult = gesture.recognizeForVideo(video, ts) } catch { /* skip */ }
    if (gestureResult) lastGestureRef.current = gestureResult

    // 상호작용 적용 — 각 트랙의 effectVisibility 갱신
    applyInteraction(tracksRef.current, lastGestureRef.current, refs.interaction.current, vw, vh, ts)

    // ─── Silhouette 모드일 때만 segmenter 실행 ───
    if (refs.debug.current && refs.shape.current === 'silhouette') {
      try {
        const seg = segmenter.segmentForVideo(video, ts)
        if (seg) lastSegMaskRef.current = seg
      } catch { /* skip */ }
    }

    // ─── 효과 렌더 ───
    if (refs.showOverlay.current) {
      const fxType = refs.effect.current
      for (const t of tracksRef.current) {
        if (t.effectVisibility <= 0.01) continue
        drawEffect(ctx, t, vw, vh, ts, mirrored, fxType)
      }
    }

    // ─── 디버그 오버레이 ───
    if (refs.debug.current) {
      if (refs.shape.current === 'silhouette' && lastSegMaskRef.current) {
        drawSilhouette(ctx, lastSegMaskRef.current, vw, vh, mirrored)
      }
      for (const t of tracksRef.current) {
        if (refs.shape.current === 'box') drawDebugBox(ctx, t, vw, ts, mirrored)
        drawDebugLabel(ctx, t, vw, ts, mirrored)
      }
    }
  }

  return (
    <div style={containerStyle}>
      <video ref={videoRef} playsInline muted style={{ display: 'none' }} />
      <canvas ref={canvasRef} style={canvasStyle} />

      <ControlBar
        debug={debug}
        status={status}
        errorMsg={errorMsg}
        fps={fps}
        trackCount={trackCount}
        mirror={mirror}
        showOverlay={showOverlay}
        effect={effect}
        shape={shape}
        interaction={interaction}
        setDebug={setDebug}
        setMirror={setMirror}
        setShowOverlay={setShowOverlay}
        setEffect={setEffect}
        setShape={setShape}
        setInteraction={setInteraction}
      />
    </div>
  )
}

// ─── 트래커 ─────────────────────────────────────────────

function updateTracks(
  tracks: Track[],
  detections: { bbox: BBox; score: number }[],
  now: number,
  nextIdRef: { current: number },
) {
  const used = new Set<number>()
  for (const track of tracks) {
    let bestIdx = -1
    let bestIoU = TRACK_MATCH_IOU
    for (let i = 0; i < detections.length; i++) {
      if (used.has(i)) continue
      const iou = iouOf(track.bbox, detections[i].bbox)
      if (iou > bestIoU) { bestIoU = iou; bestIdx = i }
    }
    if (bestIdx >= 0) {
      const det = detections[bestIdx]
      track.bbox = {
        x: lerp(track.bbox.x, det.bbox.x, POSITION_ALPHA),
        y: lerp(track.bbox.y, det.bbox.y, POSITION_ALPHA),
        w: lerp(track.bbox.w, det.bbox.w, POSITION_ALPHA),
        h: lerp(track.bbox.h, det.bbox.h, POSITION_ALPHA),
      }
      track.score = lerp(track.score, det.score, SCORE_ALPHA)
      track.lastSeenAt = now
      used.add(bestIdx)
    }
  }
  for (let i = 0; i < detections.length; i++) {
    if (used.has(i)) continue
    const det = detections[i]
    tracks.push({
      id: nextIdRef.current++,
      bbox: det.bbox,
      score: det.score,
      firstSeenAt: now,
      lastSeenAt: now,
      effectVisibility: 0,
      lastCenter: { x: det.bbox.x + det.bbox.w / 2, y: det.bbox.y + det.bbox.h / 2, ts: now },
      stillSince: null,
      prevTopY: det.bbox.y,
      jumpBoost: 0,
    })
  }
  for (let i = tracks.length - 1; i >= 0; i--) {
    if (now - tracks[i].lastSeenAt > TRACK_TIMEOUT_MS) tracks.splice(i, 1)
  }
}

function iouOf(a: BBox, b: BBox): number {
  const x1 = Math.max(a.x, b.x), y1 = Math.max(a.y, b.y)
  const x2 = Math.min(a.x + a.w, b.x + b.w), y2 = Math.min(a.y + a.h, b.y + b.h)
  if (x2 <= x1 || y2 <= y1) return 0
  const inter = (x2 - x1) * (y2 - y1)
  const union = a.w * a.h + b.w * b.h - inter
  return union > 0 ? inter / union : 0
}

function lerp(a: number, b: number, t: number): number { return a + (b - a) * t }
function clamp(v: number, lo: number, hi: number): number { return Math.max(lo, Math.min(hi, v)) }
function easeOut(t: number): number { return 1 - Math.pow(1 - t, 3) }

// ─── 상호작용 ───────────────────────────────────────────

function applyInteraction(
  tracks: Track[],
  gesture: GestureRecognizerResult | null,
  mode: InteractionMode,
  vw: number,
  vh: number,
  now: number,
) {
  // 등장 페이드인 + 모드별 가시성 타겟 계산
  for (const t of tracks) {
    const age = now - t.firstSeenAt
    const entryFactor = clamp(age / ENTRY_ANIM_MS, 0, 1)
    const entryVis = easeOut(entryFactor)

    let targetVis = entryVis

    // 모션 분석
    const cx = t.bbox.x + t.bbox.w / 2
    const cy = t.bbox.y + t.bbox.h / 2
    const dt = now - t.lastCenter.ts
    const moveDist = Math.hypot(cx - t.lastCenter.x, cy - t.lastCenter.y)
    if (moveDist > STILLNESS_MOVE_PX) t.stillSince = null
    else if (t.stillSince === null) t.stillSince = now
    t.lastCenter = { x: cx, y: cy, ts: now }

    // 점프 (위로 빠르게 이동)
    if (t.prevTopY !== null && dt > 0) {
      const dy = t.prevTopY - t.bbox.y // 양수 = 위로 이동
      const speed = dy / dt * 1000     // px/sec
      if (speed > 400) t.jumpBoost = Math.min(1, t.jumpBoost + 0.5)
      else t.jumpBoost = Math.max(0, t.jumpBoost - 0.05)
    }
    t.prevTopY = t.bbox.y

    if (mode === 'palm-hide' && gesture) {
      // 손바닥(Open_Palm)이 이 트랙 가까이 있으면 효과 숨김
      const palmNear = palmCloseToTrack(gesture, t.bbox, vw, vh)
      if (palmNear) targetVis = 0
    }
    if (mode === 'stillness-boost') {
      // 정지하면 효과 강해짐(밝기), 움직이면 약함
      if (t.stillSince !== null && now - t.stillSince > STILLNESS_DWELL_MS) targetVis = entryVis * 1.0
      else targetVis = entryVis * 0.45
    }
    // jump-grow 모드는 가시성은 그대로(entry), 크기만 변화 → drawEffect에서 jumpBoost 사용

    // 부드럽게 transition
    const k = mode === 'palm-hide' ? 16 : 8 // palm은 더 즉각, stillness는 부드럽게
    const dtSec = 1 / 60
    t.effectVisibility = approach(t.effectVisibility, targetVis, dtSec * k)
  }
}

function approach(curr: number, target: number, step: number): number {
  if (Math.abs(target - curr) < 0.001) return target
  return curr + (target - curr) * clamp(step, 0, 1)
}

function palmCloseToTrack(g: GestureRecognizerResult, bbox: BBox, vw: number, vh: number): boolean {
  if (!g.gestures || !g.landmarks) return false
  for (let i = 0; i < g.gestures.length; i++) {
    const cat = g.gestures[i]?.[0]
    if (!cat) continue
    if (cat.categoryName !== 'Open_Palm') continue
    const lm = g.landmarks[i]
    if (!lm || !lm.length) continue
    // 손의 중심 (wrist + middle finger MCP 평균)
    const wrist = lm[0]
    const mid = lm[9] ?? lm[0]
    const hx = ((wrist.x + mid.x) / 2) * vw
    const hy = ((wrist.y + mid.y) / 2) * vh
    const cx = bbox.x + bbox.w / 2
    const cy = bbox.y + bbox.h / 2
    const dist = Math.hypot(hx - cx, hy - cy)
    if (dist < bbox.w * PALM_PROXIMITY_RATIO) return true
  }
  return false
}

// ─── 효과 ───────────────────────────────────────────────

function drawEffect(
  ctx: CanvasRenderingContext2D,
  t: Track,
  vw: number,
  vh: number,
  ts: number,
  mirrored: boolean,
  fx: EffectType,
) {
  const headSize = Math.max(40, t.bbox.w * 0.35)
  let cx = t.bbox.x + t.bbox.w / 2
  const cy = t.bbox.y + headSize * 0.5
  if (mirrored) cx = vw - cx

  const vis = t.effectVisibility
  const jumpScale = 1 + t.jumpBoost * 0.6
  const scale = lerp(0.6, 1.0, vis) * jumpScale
  const alpha = vis

  ctx.save()
  ctx.globalAlpha *= alpha
  ctx.globalCompositeOperation = 'screen'

  switch (fx) {
    case 'halo':
      drawHalo(ctx, cx, cy, headSize * 1.1 * scale)
      break
    case 'sequence':
      drawSequenceFrame(ctx, cx, cy, headSize * 1.3 * scale, ts, t.id)
      break
    case 'ring':
      drawRing(ctx, cx, cy, headSize * 1.2 * scale, ts, t.id)
      break
    case 'particles':
      drawParticles(ctx, cx, cy, headSize * 1.4 * scale, ts, t.id)
      break
  }
  ctx.restore()

  void vh
}

function drawHalo(ctx: CanvasRenderingContext2D, cx: number, cy: number, r: number) {
  const grad = ctx.createRadialGradient(cx, cy, 0, cx, cy, r)
  grad.addColorStop(0, 'rgba(255, 230, 130, 0.95)')
  grad.addColorStop(0.45, 'rgba(255, 170, 60, 0.5)')
  grad.addColorStop(1, 'rgba(255, 80, 0, 0)')
  ctx.fillStyle = grad
  ctx.beginPath()
  ctx.arc(cx, cy, r, 0, Math.PI * 2)
  ctx.fill()
}

// 코드로 그리는 샘플 시퀀스 — 회전+펄스. 추후 실제 PNG 시퀀스로 교체 가능
function drawSequenceFrame(ctx: CanvasRenderingContext2D, cx: number, cy: number, r: number, ts: number, seed: number) {
  const FRAMES = 24
  const PERIOD_MS = 1600
  const frame = Math.floor(((ts + seed * 137) % PERIOD_MS) / PERIOD_MS * FRAMES)
  const phase = (frame / FRAMES) * Math.PI * 2

  // 외곽 글로우 펄스
  const pulse = 0.85 + 0.15 * Math.sin(phase * 2)
  const grad = ctx.createRadialGradient(cx, cy, 0, cx, cy, r * pulse)
  grad.addColorStop(0, 'rgba(180, 230, 255, 0.85)')
  grad.addColorStop(0.5, 'rgba(120, 180, 255, 0.5)')
  grad.addColorStop(1, 'rgba(80, 120, 255, 0)')
  ctx.fillStyle = grad
  ctx.beginPath()
  ctx.arc(cx, cy, r * pulse, 0, Math.PI * 2)
  ctx.fill()

  // 회전하는 광원 6개
  const N = 6
  for (let i = 0; i < N; i++) {
    const a = phase + (i / N) * Math.PI * 2
    const px = cx + Math.cos(a) * r * 0.7
    const py = cy + Math.sin(a) * r * 0.7
    const sr = r * 0.18 * (0.7 + 0.3 * Math.sin(phase * 3 + i))
    const g2 = ctx.createRadialGradient(px, py, 0, px, py, sr)
    g2.addColorStop(0, 'rgba(255, 250, 200, 0.95)')
    g2.addColorStop(1, 'rgba(255, 200, 100, 0)')
    ctx.fillStyle = g2
    ctx.beginPath()
    ctx.arc(px, py, sr, 0, Math.PI * 2)
    ctx.fill()
  }
}

function drawRing(ctx: CanvasRenderingContext2D, cx: number, cy: number, r: number, ts: number, seed: number) {
  const rot = (ts / 1200 + seed * 0.7) % (Math.PI * 2)
  ctx.lineWidth = Math.max(3, r * 0.06)
  for (let i = 0; i < 3; i++) {
    const a0 = rot + (i / 3) * Math.PI * 2
    const a1 = a0 + Math.PI * 0.5
    ctx.strokeStyle = `hsla(${(seed * 67 + i * 30) % 360}, 90%, 70%, 0.85)`
    ctx.beginPath()
    ctx.arc(cx, cy, r * (0.95 - i * 0.08), a0, a1)
    ctx.stroke()
  }
}

function drawParticles(ctx: CanvasRenderingContext2D, cx: number, cy: number, r: number, ts: number, seed: number) {
  const N = 14
  for (let i = 0; i < N; i++) {
    const base = (i / N) * Math.PI * 2
    const a = base + ts / 900 * (i % 2 === 0 ? 1 : -1)
    const radius = r * (0.5 + 0.5 * (0.6 + 0.4 * Math.sin(ts / 700 + i)))
    const px = cx + Math.cos(a) * radius
    const py = cy + Math.sin(a) * radius
    const sr = r * 0.08
    const grad = ctx.createRadialGradient(px, py, 0, px, py, sr)
    grad.addColorStop(0, `hsla(${(seed * 23 + i * 25 + ts / 30) % 360}, 95%, 75%, 0.9)`)
    grad.addColorStop(1, 'rgba(255,255,255,0)')
    ctx.fillStyle = grad
    ctx.beginPath()
    ctx.arc(px, py, sr, 0, Math.PI * 2)
    ctx.fill()
  }
  void seed
}

// ─── 디버그 시각화 ──────────────────────────────────────

function drawDebugBox(ctx: CanvasRenderingContext2D, t: Track, vw: number, _now: number, mirrored: boolean) {
  let x = t.bbox.x
  if (mirrored) x = vw - t.bbox.x - t.bbox.w
  const { y, w, h } = t.bbox
  const color = colorForId(t.id)
  ctx.save()
  ctx.strokeStyle = color
  ctx.lineWidth = 3
  ctx.strokeRect(x, y, w, h)
  ctx.restore()
}

function drawDebugLabel(ctx: CanvasRenderingContext2D, t: Track, vw: number, now: number, mirrored: boolean) {
  let x = t.bbox.x
  if (mirrored) x = vw - t.bbox.x - t.bbox.w
  const { y } = t.bbox
  const dwellSec = (now - t.firstSeenAt) / 1000
  const dwellStr = dwellSec < 10 ? dwellSec.toFixed(1) : Math.round(dwellSec).toString()
  const color = colorForId(t.id)
  const label = `#${t.id}   ${dwellStr}s   ${(t.score * 100).toFixed(0)}%`

  ctx.save()
  ctx.font = 'bold 18px ui-monospace, Menlo, monospace'
  const textW = ctx.measureText(label).width
  const padX = 10
  const labelH = 28
  const labelY = Math.max(0, y - labelH - 2)
  ctx.fillStyle = 'rgba(0, 0, 0, 0.85)'
  ctx.fillRect(x, labelY, textW + padX * 2, labelH)
  ctx.strokeStyle = color
  ctx.lineWidth = 2
  ctx.strokeRect(x, labelY, textW + padX * 2, labelH)
  ctx.fillStyle = color
  ctx.textBaseline = 'middle'
  ctx.fillText(label, x + padX, labelY + labelH / 2 + 1)
  ctx.restore()
}

function drawSilhouette(ctx: CanvasRenderingContext2D, seg: ImageSegmenterResult, vw: number, vh: number, mirrored: boolean) {
  const mask = seg.categoryMask
  if (!mask) return
  const w = mask.width
  const h = mask.height
  const data = mask.getAsUint8Array()
  // 마스크를 캔버스로 합성
  const off = document.createElement('canvas')
  off.width = w; off.height = h
  const offCtx = off.getContext('2d')!
  const img = offCtx.createImageData(w, h)
  for (let i = 0; i < data.length; i++) {
    const v = data[i] // 0 = background, !=0 = foreground (selfie_segmenter)
    const o = i * 4
    if (v === 0) {
      img.data[o] = 0; img.data[o + 1] = 255; img.data[o + 2] = 200; img.data[o + 3] = 0
    } else {
      img.data[o] = 0; img.data[o + 1] = 255; img.data[o + 2] = 200; img.data[o + 3] = 110
    }
  }
  offCtx.putImageData(img, 0, 0)

  ctx.save()
  if (mirrored) { ctx.translate(vw, 0); ctx.scale(-1, 1) }
  ctx.globalCompositeOperation = 'screen'
  ctx.drawImage(off, 0, 0, vw, vh)
  ctx.restore()
}

const COLORS = ['#7ee', '#ff7', '#f7f', '#7f7', '#f77', '#77f', '#fa7', '#7fa', '#a7f', '#f7a']
function colorForId(id: number): string { return COLORS[id % COLORS.length] }

// ─── 컨트롤바 (우상단) ──────────────────────────────────

type Setters = {
  setDebug: React.Dispatch<React.SetStateAction<boolean>>
  setMirror: React.Dispatch<React.SetStateAction<boolean>>
  setShowOverlay: React.Dispatch<React.SetStateAction<boolean>>
  setEffect: React.Dispatch<React.SetStateAction<EffectType>>
  setShape: React.Dispatch<React.SetStateAction<ShapeMode>>
  setInteraction: React.Dispatch<React.SetStateAction<InteractionMode>>
}

function ControlBar(props: {
  debug: boolean
  status: Status
  errorMsg: string
  fps: number
  trackCount: number
  mirror: boolean
  showOverlay: boolean
  effect: EffectType
  shape: ShapeMode
  interaction: InteractionMode
} & Setters) {
  const { debug, status, errorMsg, fps, trackCount, mirror, showOverlay, effect, shape, interaction,
    setDebug, setMirror, setShowOverlay, setEffect, setShape, setInteraction } = props

  return (
    <div style={barWrapStyle}>
      <button
        type="button"
        title="디버그 토글 (D)"
        onClick={() => setDebug((d) => !d)}
        style={dotStyle(debug)}
      />
      {debug && (
        <div style={panelStyle}>
          <StatusLine status={status} errorMsg={errorMsg} fps={fps} trackCount={trackCount} />
          <Row label="모드">
            <Toggle on={mirror} onClick={() => setMirror((v) => !v)}>거울 <K>M</K></Toggle>
            <Toggle on={showOverlay} onClick={() => setShowOverlay((v) => !v)}>효과 <K>O</K></Toggle>
            <Toggle on={false} onClick={toggleFullscreen}>풀스크린 <K>F</K></Toggle>
          </Row>
          <Row label="표시">
            <Toggle on={shape === 'box'} onClick={() => setShape('box')}>박스 <K>B</K></Toggle>
            <Toggle on={shape === 'silhouette'} onClick={() => setShape('silhouette')}>윤곽 <K>S</K></Toggle>
          </Row>
          <Row label="효과">
            <Toggle on={effect === 'halo'} onClick={() => setEffect('halo')}>halo <K>1</K></Toggle>
            <Toggle on={effect === 'sequence'} onClick={() => setEffect('sequence')}>시퀀스 <K>2</K></Toggle>
            <Toggle on={effect === 'ring'} onClick={() => setEffect('ring')}>ring <K>3</K></Toggle>
            <Toggle on={effect === 'particles'} onClick={() => setEffect('particles')}>파티클 <K>4</K></Toggle>
          </Row>
          <Row label="상호작용">
            <Toggle on={interaction === 'none'} onClick={() => setInteraction('none')}>없음 <K>N</K></Toggle>
            <Toggle on={interaction === 'palm-hide'} onClick={() => setInteraction('palm-hide')}>손바닥숨김 <K>P</K></Toggle>
            <Toggle on={interaction === 'stillness-boost'} onClick={() => setInteraction('stillness-boost')}>정지강화 <K>T</K></Toggle>
            <Toggle on={interaction === 'jump-grow'} onClick={() => setInteraction('jump-grow')}>점프확대 <K>J</K></Toggle>
          </Row>
        </div>
      )}
    </div>
  )
}

function StatusLine(props: { status: Status; errorMsg: string; fps: number; trackCount: number }) {
  const { status, errorMsg, fps, trackCount } = props
  const label: Record<Status, string> = {
    idle: '⚪ idle',
    'loading-model': '⏳ 모델 로딩…',
    'requesting-camera': '📷 카메라…',
    running: '🟢 running',
    error: '🔴 error',
  }
  const fpsColor = fps >= 50 ? '#7ee' : fps >= 30 ? '#ff7' : '#f77'
  return (
    <div style={statusLineStyle}>
      <span>{label[status]}</span>
      {status === 'running' && (
        <>
          <Sep />
          <span style={{ color: fpsColor }}>{fps} fps</span>
          <Sep />
          <span>👥 {trackCount}명</span>
        </>
      )}
      {status === 'error' && <span style={{ color: '#f77' }}>· {errorMsg}</span>}
    </div>
  )
}

function Row({ label, children }: { label: string; children: React.ReactNode }) {
  return (
    <div style={rowStyle}>
      <span style={rowLabelStyle}>{label}</span>
      <div style={rowChildrenStyle}>{children}</div>
    </div>
  )
}

function Toggle({ on, onClick, children }: { on: boolean; onClick: () => void; children: React.ReactNode }) {
  return (
    <button type="button" onClick={onClick} style={toggleStyle(on)}>
      {children}
    </button>
  )
}

function Sep() { return <span style={{ opacity: 0.4 }}>·</span> }

function K({ children }: { children: React.ReactNode }) {
  return <kbd style={kbdStyle}>{children}</kbd>
}

function toggleFullscreen() {
  if (!document.fullscreenElement) document.documentElement.requestFullscreen().catch(() => {})
  else document.exitFullscreen().catch(() => {})
}

// ─── 스타일 ─────────────────────────────────────────────

const containerStyle: React.CSSProperties = {
  position: 'fixed',
  inset: 0,
  background: '#000',
  overflow: 'hidden',
}

const canvasStyle: React.CSSProperties = {
  width: '100%',
  height: '100%',
  objectFit: 'cover',
  display: 'block',
}

const barWrapStyle: React.CSSProperties = {
  position: 'fixed',
  top: 14,
  right: 14,
  zIndex: 10,
  display: 'flex',
  flexDirection: 'column',
  alignItems: 'flex-end',
  gap: 10,
  maxWidth: 'calc(100vw - 28px)',
  fontFamily: 'ui-monospace, SFMono-Regular, Menlo, monospace',
  color: '#fff',
  textShadow: '0 1px 2px rgba(0,0,0,0.8)',
}

const dotStyle = (on: boolean): React.CSSProperties => ({
  width: 16,
  height: 16,
  borderRadius: '50%',
  border: '1px solid rgba(255,255,255,0.6)',
  background: on ? '#7ee' : 'rgba(255,255,255,0.18)',
  cursor: 'pointer',
  padding: 0,
  boxShadow: on ? '0 0 12px rgba(126,238,238,0.7)' : 'none',
})

const panelStyle: React.CSSProperties = {
  display: 'flex',
  flexDirection: 'column',
  gap: 8,
  fontSize: 13,
  maxWidth: 'min(420px, calc(100vw - 28px))',
}

const statusLineStyle: React.CSSProperties = {
  display: 'flex',
  alignItems: 'center',
  gap: 8,
  fontWeight: 600,
  fontSize: 13,
}

const rowStyle: React.CSSProperties = {
  display: 'flex',
  alignItems: 'center',
  gap: 8,
  flexWrap: 'wrap',
  justifyContent: 'flex-end',
}

const rowLabelStyle: React.CSSProperties = {
  fontSize: 11,
  opacity: 0.7,
  letterSpacing: 0.5,
  textTransform: 'uppercase',
}

const rowChildrenStyle: React.CSSProperties = {
  display: 'flex',
  flexWrap: 'wrap',
  gap: 6,
  justifyContent: 'flex-end',
}

const toggleStyle = (on: boolean): React.CSSProperties => ({
  background: on ? 'rgba(126,238,238,0.18)' : 'rgba(255,255,255,0.04)',
  border: `1px solid ${on ? 'rgba(126,238,238,0.7)' : 'rgba(255,255,255,0.25)'}`,
  color: '#fff',
  padding: '4px 8px',
  borderRadius: 6,
  cursor: 'pointer',
  fontSize: 12,
  fontFamily: 'inherit',
  display: 'inline-flex',
  alignItems: 'center',
  gap: 5,
  textShadow: '0 1px 2px rgba(0,0,0,0.6)',
})

const kbdStyle: React.CSSProperties = {
  background: 'rgba(0,0,0,0.5)',
  border: '1px solid rgba(255,255,255,0.25)',
  borderRadius: 3,
  padding: '0 4px',
  fontSize: 10,
  fontFamily: 'inherit',
  textShadow: 'none',
}
