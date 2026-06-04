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
const ENTRY_ANIM_MS = 600
const SPOTIFY_ENTRY_MS = 700
const PALM_PROXIMITY_RATIO = 0.6
const STILLNESS_DWELL_MS = 1500
const STILLNESS_MOVE_PX = 25
const EVADE_INFLUENCE_PX = 200
const EVADE_MAX_OFFSET_PX = 90
const FS_UI_HIDE_MS = 3000

type Status = 'idle' | 'loading-model' | 'requesting-camera' | 'running' | 'error'
type BBox = { x: number; y: number; w: number; h: number }
type EffectType = 'none' | 'spotify' | 'halo' | 'sequence' | 'ring' | 'particles'
type ShapeMode = 'none' | 'box' | 'silhouette-bg' | 'silhouette-fg'
type InteractionMode = 'none' | 'palm-hide' | 'stillness-boost' | 'jump-grow' | 'evade'

type Track = {
  id: number
  bbox: BBox
  score: number
  firstSeenAt: number
  effectStartedAt: number
  lastSeenAt: number
  effectVisibility: number
  lastCenter: { x: number; y: number; ts: number }
  stillSince: number | null
  prevTopY: number | null
  jumpBoost: number
  evadeOffset: { x: number; y: number }
}

export default function App() {
  const videoRef = useRef<HTMLVideoElement>(null)
  const canvasRef = useRef<HTMLCanvasElement>(null)
  const detectorRef = useRef<ObjectDetector | null>(null)
  const gestureRef = useRef<GestureRecognizer | null>(null)
  const segmenterRef = useRef<ImageSegmenter | null>(null)

  const [status, setStatus] = useState<Status>('idle')
  const [errorMsg, setErrorMsg] = useState('')
  const [panelOpen, setPanelOpen] = useState(false)
  const [fps, setFps] = useState(0)
  const [trackCount, setTrackCount] = useState(0)
  const [effect, setEffectState] = useState<EffectType>('none')
  const [shape, setShape] = useState<ShapeMode>('none')
  const [interaction, setInteraction] = useState<InteractionMode>('none')
  const [statusVisible, setStatusVisible] = useState(false)
  const [fsUiVisible, setFsUiVisible] = useState(true)
  const [isFullscreen, setIsFullscreen] = useState(false)

  // 거울은 항상 ON으로 고정
  const mirrorRef = useRef(true)
  const showOverlayRef = useRef(true)
  const refs = {
    effect: useRef(effect),
    shape: useRef(shape),
    interaction: useRef(interaction),
  }
  useEffect(() => { refs.effect.current = effect }, [effect])
  useEffect(() => { refs.shape.current = shape }, [shape])
  useEffect(() => { refs.interaction.current = interaction }, [interaction])

  const tracksRef = useRef<Track[]>([])
  const nextIdRef = useRef(1)
  const lastGestureRef = useRef<GestureRecognizerResult | null>(null)
  const lastSegMaskRef = useRef<ImageSegmenterResult | null>(null)

  // 효과 변경 시 모든 트랙의 등장 애니 재시작
  function setEffect(next: EffectType) {
    const now = performance.now()
    for (const t of tracksRef.current) t.effectStartedAt = now
    setEffectState(next)
  }

  // 마우스/터치 움직임 감지 → 풀스크린 UI 자동 숨김
  useEffect(() => {
    let timer: ReturnType<typeof setTimeout> | null = null
    function show() {
      setFsUiVisible(true)
      if (timer) clearTimeout(timer)
      timer = setTimeout(() => setFsUiVisible(false), FS_UI_HIDE_MS)
    }
    window.addEventListener('mousemove', show)
    window.addEventListener('touchstart', show)
    window.addEventListener('pointerdown', show)
    show()
    return () => {
      window.removeEventListener('mousemove', show)
      window.removeEventListener('touchstart', show)
      window.removeEventListener('pointerdown', show)
      if (timer) clearTimeout(timer)
    }
  }, [])

  // 풀스크린 상태 추적
  useEffect(() => {
    function onChange() { setIsFullscreen(!!document.fullscreenElement) }
    document.addEventListener('fullscreenchange', onChange)
    return () => document.removeEventListener('fullscreenchange', onChange)
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
        if (cancelled) { stream.getTracks().forEach((t) => t.stop()); return }
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
    const mirrored = mirrorRef.current

    ctx.save()
    if (mirrored) { ctx.translate(vw, 0); ctx.scale(-1, 1) }
    ctx.drawImage(video, 0, 0, vw, vh)
    ctx.restore()

    // ─── Person detection
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

    // ─── Gesture
    let gestureResult: GestureRecognizerResult | undefined
    try { gestureResult = gesture.recognizeForVideo(video, ts) } catch { /* skip */ }
    if (gestureResult) lastGestureRef.current = gestureResult

    applyInteraction(tracksRef.current, lastGestureRef.current, refs.interaction.current, vw, vh, ts, refs.effect.current)

    // ─── 표시
    const shapeMode = refs.shape.current
    if (shapeMode === 'silhouette-bg' || shapeMode === 'silhouette-fg') {
      try {
        const seg = segmenter.segmentForVideo(video, ts)
        if (seg) lastSegMaskRef.current = seg
      } catch { /* skip */ }
      if (lastSegMaskRef.current) {
        drawSilhouette(ctx, lastSegMaskRef.current, vw, vh, mirrored, shapeMode === 'silhouette-bg' ? 'bg' : 'fg')
      }
    }
    if (shapeMode === 'box') {
      for (const t of tracksRef.current) drawBBox(ctx, t, vw, mirrored)
    }
    if (shapeMode !== 'none') {
      for (const t of tracksRef.current) drawHeadLabel(ctx, t, vw, mirrored)
    }

    // ─── 효과 렌더
    if (showOverlayRef.current && refs.effect.current !== 'none') {
      const fxType = refs.effect.current
      for (const t of tracksRef.current) {
        if (t.effectVisibility <= 0.01) continue
        drawEffect(ctx, t, vw, ts, mirrored, fxType)
      }
    }
  }

  return (
    <div style={containerStyle}>
      <video ref={videoRef} playsInline muted style={{ display: 'none' }} />
      <canvas ref={canvasRef} style={canvasStyle} />

      <button
        type="button"
        onClick={() => setStatusVisible((v) => !v)}
        style={hiddenToggleStyle}
        title="상태 토글"
        aria-label="상태 토글"
      />
      {statusVisible && (
        <StatusOverlay status={status} errorMsg={errorMsg} fps={fps} trackCount={trackCount} />
      )}

      <BottomPanel
        open={panelOpen}
        toggle={() => setPanelOpen((v) => !v)}
        effect={effect}
        shape={shape}
        interaction={interaction}
        setEffect={setEffect}
        setShape={setShape}
        setInteraction={setInteraction}
      />

      {fsUiVisible && (
        <FullscreenButton isFullscreen={isFullscreen} />
      )}
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
      effectStartedAt: now,
      lastSeenAt: now,
      effectVisibility: 0,
      lastCenter: { x: det.bbox.x + det.bbox.w / 2, y: det.bbox.y + det.bbox.h / 2, ts: now },
      stillSince: null,
      prevTopY: det.bbox.y,
      jumpBoost: 0,
      evadeOffset: { x: 0, y: 0 },
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

// ─── 효과 위치 계산 (인터랙션과 렌더에서 공유) ──────────

function effectAnchor(t: Track, fx: EffectType) {
  const headSize = Math.max(40, t.bbox.w * 0.35)
  if (fx === 'spotify') {
    // 머리 우상단 옆, 얼굴 안 가리는 위치
    return {
      cx: t.bbox.x + t.bbox.w / 2 + headSize * 0.85,
      cy: t.bbox.y + headSize * 0.35,
      headSize,
    }
  }
  return {
    cx: t.bbox.x + t.bbox.w / 2,
    cy: t.bbox.y + headSize * 0.5,
    headSize,
  }
}

// ─── 상호작용 ───────────────────────────────────────────

function applyInteraction(
  tracks: Track[],
  gesture: GestureRecognizerResult | null,
  mode: InteractionMode,
  vw: number,
  vh: number,
  now: number,
  effect: EffectType,
) {
  for (const t of tracks) {
    const age = now - t.firstSeenAt
    const entryFactor = clamp(age / ENTRY_ANIM_MS, 0, 1)
    const entryVis = easeOut(entryFactor)
    let targetVis = entryVis

    const cx = t.bbox.x + t.bbox.w / 2
    const cy = t.bbox.y + t.bbox.h / 2
    const dt = now - t.lastCenter.ts
    const moveDist = Math.hypot(cx - t.lastCenter.x, cy - t.lastCenter.y)
    if (moveDist > STILLNESS_MOVE_PX) t.stillSince = null
    else if (t.stillSince === null) t.stillSince = now
    t.lastCenter = { x: cx, y: cy, ts: now }

    if (t.prevTopY !== null && dt > 0) {
      const dy = t.prevTopY - t.bbox.y
      const speed = dy / dt * 1000
      if (speed > 400) t.jumpBoost = Math.min(1, t.jumpBoost + 0.5)
      else t.jumpBoost = Math.max(0, t.jumpBoost - 0.05)
    }
    t.prevTopY = t.bbox.y

    let evadeTargetX = 0, evadeTargetY = 0

    if (mode === 'palm-hide' && gesture) {
      if (palmCloseToTrack(gesture, t.bbox, vw, vh)) targetVis = 0
    } else if (mode === 'stillness-boost') {
      if (t.stillSince !== null && now - t.stillSince > STILLNESS_DWELL_MS) targetVis = entryVis
      else targetVis = entryVis * 0.45
    } else if (mode === 'evade' && gesture && gesture.landmarks) {
      const anchor = effectAnchor(t, effect)
      const fxCx = anchor.cx + t.evadeOffset.x
      const fxCy = anchor.cy + t.evadeOffset.y
      let totalX = 0, totalY = 0
      for (const lm of gesture.landmarks) {
        if (!lm?.length) continue
        const wrist = lm[0]
        const mid = lm[9] ?? lm[0]
        const hx = ((wrist.x + mid.x) / 2) * vw
        const hy = ((wrist.y + mid.y) / 2) * vh
        const dist = Math.hypot(hx - fxCx, hy - fxCy)
        if (dist < EVADE_INFLUENCE_PX) {
          const dirX = fxCx - hx
          const dirY = fxCy - hy
          const len = Math.hypot(dirX, dirY) || 1
          const strength = ((EVADE_INFLUENCE_PX - dist) / EVADE_INFLUENCE_PX) * EVADE_MAX_OFFSET_PX
          totalX += (dirX / len) * strength
          totalY += (dirY / len) * strength
        }
      }
      evadeTargetX = totalX
      evadeTargetY = totalY
    }

    const visK = mode === 'palm-hide' ? 0.27 : 0.13
    t.effectVisibility = lerp(t.effectVisibility, targetVis, visK)

    const evadeK = (evadeTargetX !== 0 || evadeTargetY !== 0) ? 0.3 : 0.12
    t.evadeOffset.x = lerp(t.evadeOffset.x, evadeTargetX, evadeK)
    t.evadeOffset.y = lerp(t.evadeOffset.y, evadeTargetY, evadeK)
  }
}

function palmCloseToTrack(g: GestureRecognizerResult, bbox: BBox, vw: number, vh: number): boolean {
  if (!g.gestures || !g.landmarks) return false
  for (let i = 0; i < g.gestures.length; i++) {
    const cat = g.gestures[i]?.[0]
    if (!cat || cat.categoryName !== 'Open_Palm') continue
    const lm = g.landmarks[i]
    if (!lm?.length) continue
    const wrist = lm[0]
    const mid = lm[9] ?? lm[0]
    const hx = ((wrist.x + mid.x) / 2) * vw
    const hy = ((wrist.y + mid.y) / 2) * vh
    const cx = bbox.x + bbox.w / 2
    const cy = bbox.y + bbox.h / 2
    if (Math.hypot(hx - cx, hy - cy) < bbox.w * PALM_PROXIMITY_RATIO) return true
  }
  return false
}

// ─── 효과 렌더 ──────────────────────────────────────────

function drawEffect(
  ctx: CanvasRenderingContext2D,
  t: Track,
  vw: number,
  ts: number,
  mirrored: boolean,
  fx: EffectType,
) {
  const anchor = effectAnchor(t, fx)
  let cx = anchor.cx + t.evadeOffset.x
  const cy = anchor.cy + t.evadeOffset.y
  if (mirrored) cx = vw - cx

  const jumpScale = 1 + t.jumpBoost * 0.6

  if (fx === 'spotify') {
    const age = ts - t.effectStartedAt
    const tt = clamp(age / SPOTIFY_ENTRY_MS, 0, 1)
    const overshoot = Math.sin(tt * Math.PI * 2.3) * Math.exp(-tt * 3.2) * 0.5
    const entryScale = clamp(easeOut(tt) + overshoot, 0, 1.5)
    const r = anchor.headSize * 0.5 * entryScale * jumpScale * t.effectVisibility
    if (r < 0.5) return
    ctx.save()
    ctx.globalAlpha *= t.effectVisibility
    drawSpotifyLogo(ctx, cx, cy, r)
    ctx.restore()
    return
  }

  const scale = lerp(0.5, 1.0, t.effectVisibility) * jumpScale

  ctx.save()
  ctx.globalAlpha *= t.effectVisibility
  ctx.globalCompositeOperation = 'screen'
  switch (fx) {
    case 'halo':      drawHalo(ctx, cx, cy, anchor.headSize * 1.1 * scale); break
    case 'sequence':  drawSequenceFrame(ctx, cx, cy, anchor.headSize * 1.3 * scale, ts, t.id); break
    case 'ring':      drawRing(ctx, cx, cy, anchor.headSize * 1.2 * scale, ts, t.id); break
    case 'particles': drawParticles(ctx, cx, cy, anchor.headSize * 1.4 * scale, ts, t.id); break
  }
  ctx.restore()
}

function drawSpotifyLogo(ctx: CanvasRenderingContext2D, cx: number, cy: number, r: number) {
  if (r < 1) return
  // 그림자
  ctx.shadowColor = 'rgba(0,0,0,0.4)'
  ctx.shadowBlur = r * 0.4
  ctx.shadowOffsetY = r * 0.08

  ctx.fillStyle = '#1DB954'
  ctx.beginPath()
  ctx.arc(cx, cy, r, 0, Math.PI * 2)
  ctx.fill()
  ctx.shadowBlur = 0
  ctx.shadowOffsetY = 0

  ctx.strokeStyle = '#fff'
  ctx.lineCap = 'round'
  const bars = [
    { yo: -r * 0.30, rad: r * 0.78, lw: r * 0.18 },
    { yo: -r * 0.08, rad: r * 0.60, lw: r * 0.15 },
    { yo:  r * 0.14, rad: r * 0.42, lw: r * 0.12 },
  ]
  for (const b of bars) {
    ctx.lineWidth = b.lw
    ctx.beginPath()
    const yCenter = cy + b.yo + b.rad * 0.55
    ctx.arc(cx, yCenter, b.rad, Math.PI * 1.18, Math.PI * 1.82)
    ctx.stroke()
  }
}

function drawHalo(ctx: CanvasRenderingContext2D, cx: number, cy: number, r: number) {
  const grad = ctx.createRadialGradient(cx, cy, 0, cx, cy, r)
  grad.addColorStop(0, 'rgba(255, 230, 130, 0.95)')
  grad.addColorStop(0.45, 'rgba(255, 170, 60, 0.5)')
  grad.addColorStop(1, 'rgba(255, 80, 0, 0)')
  ctx.fillStyle = grad
  ctx.beginPath(); ctx.arc(cx, cy, r, 0, Math.PI * 2); ctx.fill()
}

function drawSequenceFrame(ctx: CanvasRenderingContext2D, cx: number, cy: number, r: number, ts: number, seed: number) {
  const FRAMES = 24, PERIOD_MS = 1600
  const frame = Math.floor(((ts + seed * 137) % PERIOD_MS) / PERIOD_MS * FRAMES)
  const phase = (frame / FRAMES) * Math.PI * 2
  const pulse = 0.85 + 0.15 * Math.sin(phase * 2)
  const grad = ctx.createRadialGradient(cx, cy, 0, cx, cy, r * pulse)
  grad.addColorStop(0, 'rgba(180, 230, 255, 0.85)')
  grad.addColorStop(0.5, 'rgba(120, 180, 255, 0.5)')
  grad.addColorStop(1, 'rgba(80, 120, 255, 0)')
  ctx.fillStyle = grad
  ctx.beginPath(); ctx.arc(cx, cy, r * pulse, 0, Math.PI * 2); ctx.fill()

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
    ctx.beginPath(); ctx.arc(px, py, sr, 0, Math.PI * 2); ctx.fill()
  }
}

function drawRing(ctx: CanvasRenderingContext2D, cx: number, cy: number, r: number, ts: number, seed: number) {
  const rot = (ts / 1200 + seed * 0.7) % (Math.PI * 2)
  ctx.lineWidth = Math.max(3, r * 0.06)
  for (let i = 0; i < 3; i++) {
    const a0 = rot + (i / 3) * Math.PI * 2
    const a1 = a0 + Math.PI * 0.5
    ctx.strokeStyle = `hsla(${(seed * 67 + i * 30) % 360}, 90%, 70%, 0.85)`
    ctx.beginPath(); ctx.arc(cx, cy, r * (0.95 - i * 0.08), a0, a1); ctx.stroke()
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
    ctx.beginPath(); ctx.arc(px, py, sr, 0, Math.PI * 2); ctx.fill()
  }
}

// ─── 시각화: 박스 / 윤곽 / 라벨 ─────────────────────────

function drawBBox(ctx: CanvasRenderingContext2D, t: Track, vw: number, mirrored: boolean) {
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

function drawHeadLabel(ctx: CanvasRenderingContext2D, t: Track, vw: number, mirrored: boolean) {
  let cx = t.bbox.x + t.bbox.w / 2
  if (mirrored) cx = vw - cx
  const yTop = Math.max(28, t.bbox.y)
  const color = colorForId(t.id)
  const label = `#${t.id}  ${(t.score * 100).toFixed(0)}%`

  ctx.save()
  ctx.font = 'bold 18px ui-monospace, Menlo, monospace'
  ctx.textAlign = 'center'
  ctx.textBaseline = 'bottom'
  const textW = ctx.measureText(label).width
  const padX = 10
  const padY = 4
  const labelH = 26
  const bx = cx - textW / 2 - padX
  const by = yTop - labelH - 6
  ctx.fillStyle = 'rgba(0, 0, 0, 0.78)'
  ctx.fillRect(bx, by, textW + padX * 2, labelH)
  ctx.strokeStyle = color
  ctx.lineWidth = 1.5
  ctx.strokeRect(bx, by, textW + padX * 2, labelH)
  ctx.fillStyle = color
  ctx.fillText(label, cx, by + labelH - padY)
  ctx.restore()
}

function drawSilhouette(
  ctx: CanvasRenderingContext2D,
  seg: ImageSegmenterResult,
  vw: number,
  vh: number,
  mirrored: boolean,
  side: 'bg' | 'fg',
) {
  const mask = seg.categoryMask
  if (!mask) return
  const w = mask.width
  const h = mask.height
  const data = mask.getAsUint8Array()
  const off = document.createElement('canvas')
  off.width = w; off.height = h
  const offCtx = off.getContext('2d')!
  const img = offCtx.createImageData(w, h)
  const targetIsZero = side === 'bg'
  const baseR = side === 'bg' ? 0 : 255
  const baseG = side === 'bg' ? 255 : 120
  const baseB = side === 'bg' ? 200 : 60

  for (let i = 0; i < data.length; i++) {
    const v = data[i]
    const matches = targetIsZero ? (v === 0) : (v !== 0)
    const o = i * 4
    if (matches) {
      img.data[o] = baseR; img.data[o + 1] = baseG; img.data[o + 2] = baseB; img.data[o + 3] = 120
    } else {
      img.data[o] = 0; img.data[o + 1] = 0; img.data[o + 2] = 0; img.data[o + 3] = 0
    }
  }
  offCtx.putImageData(img, 0, 0)

  ctx.save()
  if (mirrored) { ctx.translate(vw, 0); ctx.scale(-1, 1) }
  ctx.globalCompositeOperation = 'screen'
  ctx.drawImage(off, 0, 0, vw, vh)
  if (side === 'fg') {
    ctx.globalCompositeOperation = 'source-over'
    ctx.shadowColor = `rgba(${baseR}, ${baseG}, ${baseB}, 0.9)`
    ctx.shadowBlur = 16
    ctx.drawImage(off, 0, 0, vw, vh)
    ctx.shadowBlur = 0
  }
  ctx.restore()
}

const COLORS = ['#7ee', '#ff7', '#f7f', '#7f7', '#f77', '#77f', '#fa7', '#7fa', '#a7f', '#f7a']
function colorForId(id: number): string { return COLORS[id % COLORS.length] }

// ─── UI: 좌상단 상태 (텍스트만, 이모지 X) ───────────────

function StatusOverlay(props: { status: Status; errorMsg: string; fps: number; trackCount: number }) {
  const { status, errorMsg, fps, trackCount } = props
  const label: Record<Status, string> = {
    idle: 'idle',
    'loading-model': 'loading model',
    'requesting-camera': 'requesting camera',
    running: 'running',
    error: 'error',
  }
  const fpsColor = fps >= 50 ? '#7ee' : fps >= 30 ? '#ff7' : '#f77'
  return (
    <div style={statusOverlayStyle}>
      <span>{label[status]}</span>
      {status === 'running' && (
        <>
          <span style={{ opacity: 0.4 }}>·</span>
          <span style={{ color: fpsColor }}>{fps} fps</span>
          <span style={{ opacity: 0.4 }}>·</span>
          <span>{trackCount} tracked</span>
        </>
      )}
      {status === 'error' && <span style={{ color: '#f77', marginLeft: 6 }}>{errorMsg}</span>}
    </div>
  )
}

// ─── UI: 하단 중앙 패널 ─────────────────────────────────

function BottomPanel(props: {
  open: boolean
  toggle: () => void
  effect: EffectType
  shape: ShapeMode
  interaction: InteractionMode
  setEffect: (e: EffectType) => void
  setShape: React.Dispatch<React.SetStateAction<ShapeMode>>
  setInteraction: React.Dispatch<React.SetStateAction<InteractionMode>>
}) {
  const { open, toggle, effect, shape, interaction, setEffect, setShape, setInteraction } = props

  return (
    <div style={bottomWrapStyle}>
      {open && (
        <div style={panelStyle}>
          <Row label="표시">
            <Toggle on={shape === 'none'} onClick={() => setShape('none')}>없음</Toggle>
            <Toggle on={shape === 'box'} onClick={() => setShape('box')}>박스</Toggle>
            <Toggle on={shape === 'silhouette-bg'} onClick={() => setShape('silhouette-bg')}>윤곽-배경</Toggle>
            <Toggle on={shape === 'silhouette-fg'} onClick={() => setShape('silhouette-fg')}>윤곽-사람</Toggle>
          </Row>
          <Row label="효과">
            <Toggle on={effect === 'none'} onClick={() => setEffect('none')}>없음</Toggle>
            <Toggle on={effect === 'spotify'} onClick={() => setEffect('spotify')}>spotify</Toggle>
            <Toggle on={effect === 'halo'} onClick={() => setEffect('halo')}>halo</Toggle>
            <Toggle on={effect === 'sequence'} onClick={() => setEffect('sequence')}>시퀀스</Toggle>
            <Toggle on={effect === 'ring'} onClick={() => setEffect('ring')}>ring</Toggle>
            <Toggle on={effect === 'particles'} onClick={() => setEffect('particles')}>파티클</Toggle>
          </Row>
          <Row label="상호작용">
            <Toggle on={interaction === 'none'} onClick={() => setInteraction('none')}>없음</Toggle>
            <Toggle on={interaction === 'palm-hide'} onClick={() => setInteraction('palm-hide')}>손바닥숨김</Toggle>
            <Toggle on={interaction === 'stillness-boost'} onClick={() => setInteraction('stillness-boost')}>정지강화</Toggle>
            <Toggle on={interaction === 'jump-grow'} onClick={() => setInteraction('jump-grow')}>점프확대</Toggle>
            <Toggle on={interaction === 'evade'} onClick={() => setInteraction('evade')}>도망</Toggle>
          </Row>
        </div>
      )}
      <button type="button" onClick={toggle} title="효과 패널" style={fabStyle(open)}>
        <SparkleIcon />
      </button>
    </div>
  )
}

// ─── UI: 풀스크린 버튼 (우하단, 자동숨김) ──────────────

function FullscreenButton({ isFullscreen }: { isFullscreen: boolean }) {
  function onClick() {
    if (isFullscreen) document.exitFullscreen().catch(() => {})
    else document.documentElement.requestFullscreen().catch(() => {})
  }
  return (
    <button type="button" onClick={onClick} style={fsBtnStyle} title={isFullscreen ? '풀스크린 종료' : '풀스크린'}>
      {isFullscreen ? <FullscreenExitIcon /> : <FullscreenEnterIcon />}
    </button>
  )
}

function SparkleIcon() {
  return (
    <svg width="22" height="22" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.2" strokeLinecap="round" strokeLinejoin="round" aria-hidden>
      <path d="M12 3 L13.5 9 L20 10.5 L13.5 12 L12 18 L10.5 12 L4 10.5 L10.5 9 Z" />
      <path d="M19 17 L19.7 19 L22 19.7 L19.7 20.4 L19 22.5 L18.3 20.4 L16 19.7 L18.3 19 Z" />
      <path d="M5 15 L5.5 16.5 L7 17 L5.5 17.5 L5 19 L4.5 17.5 L3 17 L4.5 16.5 Z" />
    </svg>
  )
}

function FullscreenEnterIcon() {
  return (
    <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.2" strokeLinecap="round" strokeLinejoin="round" aria-hidden>
      <path d="M4 9 V4 H9" />
      <path d="M20 9 V4 H15" />
      <path d="M4 15 V20 H9" />
      <path d="M20 15 V20 H15" />
    </svg>
  )
}

function FullscreenExitIcon() {
  return (
    <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.2" strokeLinecap="round" strokeLinejoin="round" aria-hidden>
      <path d="M9 4 V9 H4" />
      <path d="M15 4 V9 H20" />
      <path d="M9 20 V15 H4" />
      <path d="M15 20 V15 H20" />
    </svg>
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
  return <button type="button" onClick={onClick} style={toggleStyle(on)}>{children}</button>
}

// ─── 스타일 ─────────────────────────────────────────────

const containerStyle: React.CSSProperties = {
  position: 'fixed', inset: 0, background: '#000', overflow: 'hidden',
}

const canvasStyle: React.CSSProperties = {
  width: '100%', height: '100%', objectFit: 'cover', display: 'block',
}

const hiddenToggleStyle: React.CSSProperties = {
  position: 'fixed',
  top: 0,
  left: 0,
  width: 28,
  height: 28,
  background: 'transparent',
  border: 'none',
  cursor: 'pointer',
  padding: 0,
  zIndex: 11,
}

const statusOverlayStyle: React.CSSProperties = {
  position: 'fixed',
  top: 14,
  left: 14,
  zIndex: 10,
  display: 'flex',
  alignItems: 'center',
  gap: 8,
  fontSize: 13,
  fontWeight: 600,
  fontFamily: 'ui-monospace, SFMono-Regular, Menlo, monospace',
  color: '#fff',
  textShadow: '0 1px 3px rgba(0,0,0,0.85)',
  pointerEvents: 'none',
  maxWidth: 'calc(100vw - 28px)',
}

const bottomWrapStyle: React.CSSProperties = {
  position: 'fixed',
  left: '50%',
  bottom: 16,
  transform: 'translateX(-50%)',
  zIndex: 10,
  display: 'flex',
  flexDirection: 'column',
  alignItems: 'center',
  gap: 10,
  maxWidth: 'calc(100vw - 28px)',
  width: 'max-content',
  fontFamily: 'ui-monospace, SFMono-Regular, Menlo, monospace',
  color: '#fff',
  textShadow: '0 1px 2px rgba(0,0,0,0.85)',
}

const panelStyle: React.CSSProperties = {
  display: 'flex',
  flexDirection: 'column',
  gap: 8,
  fontSize: 13,
  maxWidth: 'min(560px, calc(100vw - 28px))',
  alignItems: 'flex-start',
}

const rowStyle: React.CSSProperties = {
  display: 'flex',
  alignItems: 'center',
  gap: 10,
  flexWrap: 'wrap',
}

const rowLabelStyle: React.CSSProperties = {
  fontSize: 11,
  opacity: 0.7,
  letterSpacing: 0.5,
  textTransform: 'uppercase',
  minWidth: 60,
}

const rowChildrenStyle: React.CSSProperties = {
  display: 'flex',
  flexWrap: 'wrap',
  gap: 6,
}

const toggleStyle = (on: boolean): React.CSSProperties => ({
  background: on ? 'rgba(126,238,238,0.18)' : 'rgba(255,255,255,0.04)',
  border: `1px solid ${on ? 'rgba(126,238,238,0.7)' : 'rgba(255,255,255,0.25)'}`,
  color: '#fff',
  padding: '4px 10px',
  borderRadius: 6,
  cursor: 'pointer',
  fontSize: 12,
  fontFamily: 'inherit',
  textShadow: '0 1px 2px rgba(0,0,0,0.6)',
})

const fabStyle = (open: boolean): React.CSSProperties => ({
  width: 48,
  height: 48,
  borderRadius: '50%',
  border: `1px solid ${open ? 'rgba(126,238,238,0.7)' : 'rgba(255,255,255,0.4)'}`,
  background: open ? 'rgba(126,238,238,0.18)' : 'rgba(0,0,0,0.55)',
  backdropFilter: 'blur(8px)',
  color: '#fff',
  cursor: 'pointer',
  padding: 0,
  display: 'flex',
  alignItems: 'center',
  justifyContent: 'center',
  boxShadow: open ? '0 0 16px rgba(126,238,238,0.5)' : '0 2px 8px rgba(0,0,0,0.5)',
})

const fsBtnStyle: React.CSSProperties = {
  position: 'fixed',
  right: 16,
  bottom: 16,
  zIndex: 11,
  width: 42,
  height: 42,
  borderRadius: '50%',
  border: '1px solid rgba(255,255,255,0.4)',
  background: 'rgba(0,0,0,0.55)',
  backdropFilter: 'blur(8px)',
  color: '#fff',
  cursor: 'pointer',
  padding: 0,
  display: 'flex',
  alignItems: 'center',
  justifyContent: 'center',
  boxShadow: '0 2px 8px rgba(0,0,0,0.5)',
  transition: 'opacity 200ms',
}
