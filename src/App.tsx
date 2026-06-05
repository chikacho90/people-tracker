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
// selfie_multiclass: 0=background, 1=hair, 2=body-skin, 3=face-skin, 4=clothes, 5=others
// (단일 셀카 선호하지만 셀피세그멘터보다 의자/가구를 person으로 잡는 빈도 낮음)
const SEG_MODEL = 'https://storage.googleapis.com/mediapipe-models/image_segmenter/selfie_multiclass_256x256/float32/1/selfie_multiclass_256x256.tflite'

const MAX_DETECTIONS = 20
const TRACK_TIMEOUT_MS = 800
const TRACK_MATCH_IOU = 0.2
const SCORE_THRESHOLD = 0.45
const POSITION_ALPHA = 0.35
const SCORE_ALPHA = 0.15
const ENTRY_ANIM_MS = 600
const SPOTIFY_ENTRY_MS = 900
const FS_UI_HIDE_MS = 3000
const LIKE_BURST_MS = 1000
const DROP_BURST_MS = 800
const SKIP_BURST_MS = 700
const TOGETHER_RANGE_PX = 320
const HEAD_DIST_NEAR_X = 0.4    // bbox 중심 거리/평균 bbox.w 이하 시 가까움
const SPOTIFY_GREEN = '#1DB954'

// 공식 Spotify 아이콘 SVG (viewBox 168×168). 한 번만 Image로 디코딩해서 캐싱
const SPOTIFY_SVG = `<?xml version="1.0" encoding="UTF-8"?><svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 168 168"><path fill="#1DB954" d="M83.996.277C37.747.277.253 37.77.253 84.019c0 46.251 37.494 83.741 83.743 83.741 46.254 0 83.744-37.49 83.744-83.741 0-46.246-37.49-83.738-83.745-83.738l.001-.004zm38.404 120.78a5.217 5.217 0 0 1-7.18 1.73c-19.662-12.01-44.414-14.73-73.564-8.07a5.222 5.222 0 0 1-6.249-3.93 5.213 5.213 0 0 1 3.926-6.25c31.9-7.291 59.263-4.15 81.337 9.34 2.46 1.51 3.24 4.72 1.73 7.18zm10.25-22.805c-1.89 3.075-5.91 4.045-8.98 2.155-22.51-13.839-56.823-17.846-83.448-9.764-3.453 1.043-7.1-.903-8.148-4.35a6.538 6.538 0 0 1 4.354-8.143c30.413-9.228 68.222-4.758 94.072 11.127 3.07 1.89 4.04 5.91 2.15 8.976v-.001zm.88-23.744c-26.99-16.031-71.52-17.505-97.289-9.684-4.138 1.255-8.514-1.081-9.768-5.219a7.835 7.835 0 0 1 5.221-9.771c29.581-8.98 78.756-7.245 109.83 11.202a7.823 7.823 0 0 1 2.74 10.733c-2.2 3.722-7.02 4.949-10.73 2.739z"/></svg>`

const spotifyImage: HTMLImageElement = (() => {
  const img = new Image()
  img.decoding = 'async'
  img.src = 'data:image/svg+xml;utf8,' + encodeURIComponent(SPOTIFY_SVG)
  return img
})()

type Status = 'idle' | 'loading-model' | 'requesting-camera' | 'running' | 'error'
type BBox = { x: number; y: number; w: number; h: number }
type EffectType = 'none' | 'pop' | 'bounce' | 'orbit' | 'multiply' | 'breathe' | 'pulse'
type ShapeMode = 'none' | 'box' | 'silhouette-bg' | 'silhouette-fg' | 'silhouette-outline'
type InteractionMode =
  | 'none' | 'move-music' | 'volume-up' | 'tap-like' | 'listen-together'
  | 'drop-beat' | 'skip-track' | 'headphones' | 'discover' | 'group-sync'

type DiscoverLogo = { x: number; y: number; bornAt: number; ttl: number }

type Track = {
  id: number
  bbox: BBox
  score: number
  firstSeenAt: number
  effectStartedAt: number
  lastSeenAt: number
  effectVisibility: number
  lastCenter: { x: number; y: number; ts: number }
  movementLevel: number
  handsUpLevel: number
  likeBurstAt: number | null
  togetherOffset: { x: number; y: number }
  dropBurstAt: number | null
  skipBurstAt: number | null
  skipDir: -1 | 1
  headphonesActive: boolean
  discoverLogos: DiscoverLogo[]
  lastDiscoverAt: number
  groupSyncHue: number
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
  // skip-track 용 손 이전 위치 추적
  const lastHandPosRef = useRef<{ x: number; y: number; ts: number }[]>([])
  // group-sync 글로벌 hue
  const groupHueRef = useRef(0)

  function setEffect(next: EffectType) {
    const now = performance.now()
    for (const t of tracksRef.current) t.effectStartedAt = now
    setEffectState(next)
  }

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
        if (cancelled) { detector.close(); gesture.close(); segmenter.close(); return }
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
          if (dt > 0) { fpsBuf.push(1000 / dt); if (fpsBuf.length > 30) fpsBuf.shift() }
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

    let gestureResult: GestureRecognizerResult | undefined
    try { gestureResult = gesture.recognizeForVideo(video, ts) } catch { /* skip */ }
    if (gestureResult) lastGestureRef.current = gestureResult

    applyInteractions(
      tracksRef.current,
      lastGestureRef.current,
      refs.interaction.current,
      vw, vh, ts,
      lastHandPosRef.current,
      groupHueRef,
    )

    const shapeMode = refs.shape.current
    if (shapeMode === 'silhouette-bg' || shapeMode === 'silhouette-fg' || shapeMode === 'silhouette-outline') {
      try {
        const seg = segmenter.segmentForVideo(video, ts)
        if (seg) lastSegMaskRef.current = seg
      } catch { /* skip */ }
      if (lastSegMaskRef.current) {
        drawSilhouette(ctx, lastSegMaskRef.current, vw, vh, mirrored, shapeMode, tracksRef.current)
      }
    }
    if (shapeMode === 'box') {
      for (const t of tracksRef.current) drawBBox(ctx, t, vw, mirrored)
    }
    if (shapeMode !== 'none') {
      tracksRef.current.forEach((t, i) => drawHeadLabel(ctx, t, i + 1, vw, mirrored))
    }

    if (showOverlayRef.current && refs.effect.current !== 'none') {
      const fxType = refs.effect.current
      const interactionMode = refs.interaction.current
      for (const t of tracksRef.current) {
        if (t.effectVisibility <= 0.01) continue
        drawEffect(ctx, t, vw, ts, mirrored, fxType, interactionMode)
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
        title="Toggle status"
        aria-label="Toggle status"
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

      {fsUiVisible && (<FullscreenButton isFullscreen={isFullscreen} />)}
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
      movementLevel: 0,
      handsUpLevel: 0,
      likeBurstAt: null,
      togetherOffset: { x: 0, y: 0 },
      dropBurstAt: null,
      skipBurstAt: null,
      skipDir: 1,
      headphonesActive: false,
      discoverLogos: [],
      lastDiscoverAt: 0,
      groupSyncHue: 0,
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
function easeOutCubic(t: number): number { return 1 - Math.pow(1 - t, 3) }
function easeOutQuint(t: number): number { return 1 - Math.pow(1 - t, 5) }

function effectAnchor(t: Track) {
  const headSize = Math.max(40, t.bbox.w * 0.35)
  return {
    cx: t.bbox.x + t.bbox.w / 2 + headSize * 0.7,
    cy: t.bbox.y + headSize * 0.3,
    headSize,
    headCx: t.bbox.x + t.bbox.w / 2,
    headCy: t.bbox.y + headSize * 0.4,
  }
}

function handBelongsTo(hx: number, hy: number, bbox: BBox): boolean {
  // 손 위치가 사람 박스 범위 안 또는 살짝 위 있으면 그 사람의 손으로 간주
  const pad = bbox.w * 0.4
  return hx > bbox.x - pad && hx < bbox.x + bbox.w + pad &&
         hy > bbox.y - pad && hy < bbox.y + bbox.h + pad
}

// ─── 상호작용 ───────────────────────────────────────────

function applyInteractions(
  tracks: Track[],
  gesture: GestureRecognizerResult | null,
  mode: InteractionMode,
  vw: number,
  vh: number,
  now: number,
  lastHandPos: { x: number; y: number; ts: number }[],
  groupHueRef: { current: number },
) {
  // 등장 페이드 + 움직임 추적 + 기본 정리
  for (const t of tracks) {
    const age = now - t.firstSeenAt
    const entryVis = easeOutCubic(clamp(age / ENTRY_ANIM_MS, 0, 1))
    const cx = t.bbox.x + t.bbox.w / 2
    const cy = t.bbox.y + t.bbox.h / 2
    const dt = Math.max(1, now - t.lastCenter.ts)
    const moveDist = Math.hypot(cx - t.lastCenter.x, cy - t.lastCenter.y)
    t.lastCenter = { x: cx, y: cy, ts: now }

    // movement level (모드 무관 추적)
    const speedNorm = clamp((moveDist / dt) * 30, 0, 1)
    t.movementLevel = lerp(t.movementLevel, mode === 'move-music' ? speedNorm : 0, 0.1)

    // 만료된 like/drop/skip 정리
    if (t.likeBurstAt !== null && now - t.likeBurstAt > LIKE_BURST_MS) t.likeBurstAt = null
    if (t.dropBurstAt !== null && now - t.dropBurstAt > DROP_BURST_MS) t.dropBurstAt = null
    if (t.skipBurstAt !== null && now - t.skipBurstAt > SKIP_BURST_MS) t.skipBurstAt = null

    // discoverLogos 만료
    t.discoverLogos = t.discoverLogos.filter((d) => now - d.bornAt < d.ttl)

    // togetherOffset 자연 복귀 (몸 모드에서만 끌어당김)
    if (mode !== 'listen-together') {
      t.togetherOffset.x = lerp(t.togetherOffset.x, 0, 0.12)
      t.togetherOffset.y = lerp(t.togetherOffset.y, 0, 0.12)
    }
    if (mode !== 'volume-up') t.handsUpLevel = lerp(t.handsUpLevel, 0, 0.1)
    if (mode !== 'headphones') t.headphonesActive = false
    if (mode !== 'group-sync') t.groupSyncHue = lerp(t.groupSyncHue, 0, 0.05)

    t.effectVisibility = lerp(t.effectVisibility, entryVis, 0.18)
  }

  if (!gesture) return

  // hand 위치 리스트 (간단 swipe용으로 추적)
  const handsNow: { x: number; y: number; ts: number; category?: string }[] = []
  if (gesture.landmarks) {
    for (let i = 0; i < gesture.landmarks.length; i++) {
      const lm = gesture.landmarks[i]
      if (!lm?.length) continue
      const wrist = lm[0]
      const cat = gesture.gestures?.[i]?.[0]?.categoryName
      handsNow.push({ x: wrist.x * vw, y: wrist.y * vh, ts: now, category: cat })
    }
  }

  // ─── volume-up
  if (mode === 'volume-up') {
    for (const t of tracks) {
      let above = 0
      const headLine = t.bbox.y + t.bbox.w * 0.1
      for (const h of handsNow) {
        if (handBelongsTo(h.x, h.y, t.bbox) && h.y < headLine) above++
      }
      t.handsUpLevel = lerp(t.handsUpLevel, Math.min(1, above / 2), 0.15)
    }
  }

  // ─── tap-like
  if (mode === 'tap-like') {
    for (const t of tracks) {
      for (const h of handsNow) {
        if (h.category !== 'Thumb_Up') continue
        if (!handBelongsTo(h.x, h.y, t.bbox)) continue
        if (t.likeBurstAt === null || now - t.likeBurstAt > LIKE_BURST_MS) {
          t.likeBurstAt = now
        }
      }
    }
  }

  // ─── drop-beat
  if (mode === 'drop-beat') {
    for (const t of tracks) {
      for (const h of handsNow) {
        if (h.category !== 'Open_Palm') continue
        if (!handBelongsTo(h.x, h.y, t.bbox)) continue
        if (!t.dropBurstAt || now - t.dropBurstAt > 1200) t.dropBurstAt = now
      }
    }
  }

  // ─── skip-track (손 빠른 swipe)
  if (mode === 'skip-track') {
    for (const h of handsNow) {
      // 가장 가까운 이전 손 위치 찾기
      let bestDist = Infinity, bestPrev = null as typeof lastHandPos[number] | null
      for (const p of lastHandPos) {
        const d = Math.hypot(p.x - h.x, p.y - h.y)
        if (d < bestDist) { bestDist = d; bestPrev = p }
      }
      if (bestPrev && bestDist < 200) {
        const ddt = Math.max(1, now - bestPrev.ts)
        const vx = (h.x - bestPrev.x) / ddt * 1000  // px/sec
        if (Math.abs(vx) > 1000) {
          const dir: -1 | 1 = vx > 0 ? 1 : -1
          for (const t of tracks) {
            if (handBelongsTo(h.x, h.y, t.bbox)) {
              if (!t.skipBurstAt || now - t.skipBurstAt > SKIP_BURST_MS) {
                t.skipBurstAt = now
                t.skipDir = dir
              }
            }
          }
        }
      }
    }
  }
  lastHandPos.length = 0
  for (const h of handsNow) lastHandPos.push(h)

  // ─── headphones
  if (mode === 'headphones') {
    for (const t of tracks) {
      const headCx = t.bbox.x + t.bbox.w / 2
      const headCy = t.bbox.y + t.bbox.w * 0.18
      const yRange = t.bbox.w * 0.3
      let left = false, right = false
      for (const h of handsNow) {
        if (!handBelongsTo(h.x, h.y, t.bbox)) continue
        if (Math.abs(h.y - headCy) < yRange) {
          if (h.x < headCx) left = true
          if (h.x > headCx) right = true
        }
      }
      t.headphonesActive = left && right
    }
  }

  // ─── discover
  if (mode === 'discover') {
    for (const t of tracks) {
      for (let i = 0; i < (gesture.landmarks?.length ?? 0); i++) {
        const cat = gesture.gestures?.[i]?.[0]?.categoryName
        if (cat !== 'Pointing_Up') continue
        const lm = gesture.landmarks[i]
        if (!lm?.length) continue
        const wrist = lm[0]
        const tip = lm[8]
        const hx = wrist.x * vw, hy = wrist.y * vh
        if (!handBelongsTo(hx, hy, t.bbox)) continue
        if (now - t.lastDiscoverAt < 280) continue
        // 손가락 방향으로 새 로고 생성
        const dx = (tip.x - wrist.x) * vw
        const dy = (tip.y - wrist.y) * vh
        const len = Math.hypot(dx, dy) || 1
        const nx = dx / len, ny = dy / len
        const spawnX = (tip.x * vw) + nx * 60
        const spawnY = (tip.y * vh) + ny * 60
        t.discoverLogos.push({ x: spawnX, y: spawnY, bornAt: now, ttl: 1800 })
        t.lastDiscoverAt = now
      }
    }
  }

  // ─── listen-together
  if (mode === 'listen-together' && tracks.length >= 2) {
    for (const t of tracks) {
      let bestPullX = 0, bestPullY = 0, bestDist = Infinity
      const myAnchor = effectAnchor(t)
      for (const o of tracks) {
        if (o.id === t.id) continue
        const otherAnchor = effectAnchor(o)
        const dx = otherAnchor.cx - myAnchor.cx
        const dy = otherAnchor.cy - myAnchor.cy
        const dist = Math.hypot(dx, dy)
        if (dist < TOGETHER_RANGE_PX && dist < bestDist) {
          bestDist = dist
          const pull = (TOGETHER_RANGE_PX - dist) / TOGETHER_RANGE_PX * 0.5
          bestPullX = dx * pull
          bestPullY = dy * pull
        }
      }
      t.togetherOffset.x = lerp(t.togetherOffset.x, bestPullX, 0.18)
      t.togetherOffset.y = lerp(t.togetherOffset.y, bestPullY, 0.18)
    }
  }

  // ─── group-sync (모든 트랙의 첫 손이 같은 카테고리면 글로벌 hue 변화)
  if (mode === 'group-sync' && tracks.length >= 2) {
    const cats = new Set<string>()
    for (let i = 0; i < (gesture.gestures?.length ?? 0); i++) {
      const c = gesture.gestures[i]?.[0]?.categoryName
      if (c && c !== 'None') cats.add(c)
    }
    if (cats.size === 1 && handsNow.length >= 2) {
      // 모든 손이 같은 제스처
      const cat = Array.from(cats)[0]
      const hueMap: Record<string, number> = {
        'Open_Palm': 200,    // 파랑
        'Closed_Fist': 0,    // 빨강
        'Thumb_Up': 50,      // 노랑
        'Victory': 280,      // 보라
        'ILoveYou': 320,     // 핑크
        'Pointing_Up': 130,  // 초록
        'Thumb_Down': 30,    // 주황
      }
      const target = hueMap[cat] ?? 0
      groupHueRef.current = lerp(groupHueRef.current, target, 0.08)
      for (const t of tracks) t.groupSyncHue = groupHueRef.current
    } else {
      groupHueRef.current = lerp(groupHueRef.current, 0, 0.04)
    }
  }
}

// ─── 효과 렌더 ──────────────────────────────────────────

function drawEffect(
  ctx: CanvasRenderingContext2D,
  t: Track,
  vw: number,
  ts: number,
  mirrored: boolean,
  fx: EffectType,
  mode: InteractionMode,
) {
  const anchor = effectAnchor(t)
  const baseX = anchor.cx + t.togetherOffset.x
  const baseY = anchor.cy + t.togetherOffset.y
  let cx = mirrored ? vw - baseX : baseX
  let cy = baseY

  // headphones는 위치 자체를 양 귀 옆으로 변경 — 일반 효과 대신 헤드폰 표시
  if (mode === 'headphones' && t.headphonesActive) {
    let lcx = t.bbox.x + t.bbox.w * 0.2
    let rcx = t.bbox.x + t.bbox.w * 0.8
    const ecy = t.bbox.y + t.bbox.w * 0.2
    if (mirrored) { const tmp = lcx; lcx = vw - rcx; rcx = vw - tmp }
    const r = anchor.headSize * 0.12 * t.effectVisibility
    ctx.save()
    ctx.globalAlpha *= t.effectVisibility
    drawSpotifyLogo(ctx, lcx, ecy, r, 0)
    drawSpotifyLogo(ctx, rcx, ecy, r, 0)
    // 헤드밴드 라인
    ctx.strokeStyle = SPOTIFY_GREEN
    ctx.lineWidth = Math.max(2, r * 0.25)
    ctx.beginPath()
    ctx.moveTo(lcx, ecy)
    ctx.bezierCurveTo(lcx + (rcx - lcx) * 0.3, ecy - r * 1.2, rcx - (rcx - lcx) * 0.3, ecy - r * 1.2, rcx, ecy)
    ctx.stroke()
    ctx.restore()
    drawExtras(ctx, t, anchor, vw, ts, mirrored, mode)
    return
  }

  const age = ts - t.effectStartedAt
  const entryT = clamp(age / SPOTIFY_ENTRY_MS, 0, 1)
  const entryScale = easeOutCubic(entryT)
  const entryRotation = (1 - easeOutQuint(entryT)) * -0.5
  const entryAlpha = easeOutCubic(clamp(entryT * 1.4, 0, 1))

  // 효과별 base 크기 (모든 효과가 spotify 로고 작은 크기 기준)
  const baseR = anchor.headSize * 0.1

  // interaction multipliers
  const volMul = 1 + t.handsUpLevel * 1.5
  const moveMul = mode === 'move-music' ? (1 + t.movementLevel * 1.2) : 1
  const groupHue = t.groupSyncHue || 0

  ctx.save()
  ctx.globalAlpha *= entryAlpha * t.effectVisibility

  // skip-track: 효과가 그 방향으로 슬라이드 아웃 + 색 변환 후 다시
  let skipOffset = 0
  let skipColorHueShift = 0
  if (t.skipBurstAt !== null) {
    const skipT = clamp((ts - t.skipBurstAt) / SKIP_BURST_MS, 0, 1)
    const halfT = skipT < 0.5 ? skipT * 2 : (1 - skipT) * 2  // 0→1→0
    skipOffset = halfT * 200 * t.skipDir
    skipColorHueShift = skipT > 0.5 ? (skipT - 0.5) * 360 : 0
  }
  ctx.translate(skipOffset, 0)

  switch (fx) {
    case 'pop': {
      const float = Math.sin(ts / 1300 + t.id * 0.7) * baseR * 0.6
      const rot = Math.sin(ts / 2400 + t.id * 0.3) * 0.10
      const r = baseR * entryScale * volMul * moveMul
      drawColoredLogo(ctx, cx + Math.sin(ts / 1100 + t.id * 0.5) * baseR * 0.3, cy + float, r, entryRotation + rot, groupHue + skipColorHueShift)
      break
    }
    case 'bounce': {
      const bpm = 0.5 * (1 + t.movementLevel * 0.5) // 빠른 움직임 → 더 빠른 비트
      const period = 500 / bpm * 0.5
      const beatT = ((ts % period) / period)
      const bounceY = -Math.abs(Math.sin(beatT * Math.PI)) * baseR * 2.5
      const squashY = 1 - Math.abs(Math.cos(beatT * Math.PI)) * 0.15
      const r = baseR * entryScale * volMul
      ctx.save()
      ctx.translate(cx, cy + bounceY)
      ctx.scale(1 / squashY, squashY) // squash & stretch
      drawColoredLogo(ctx, 0, 0, r, entryRotation, groupHue + skipColorHueShift)
      ctx.restore()
      break
    }
    case 'orbit': {
      const N = Math.max(3, Math.round(4 + t.handsUpLevel * 4))
      const orbitR = anchor.headSize * (1.0 + t.movementLevel * 0.4) * entryScale
      const rotSpeed = (1 + t.movementLevel * 1.0) / 2000
      const r = baseR * 0.95 * entryScale * volMul
      for (let i = 0; i < N; i++) {
        const a = ts * rotSpeed + (i / N) * Math.PI * 2 + entryRotation
        const px = cx + Math.cos(a) * orbitR
        const py = cy + Math.sin(a) * orbitR * 0.6 // 약간 elliptical
        drawColoredLogo(ctx, px, py, r, a + Math.PI / 2, groupHue + skipColorHueShift)
      }
      break
    }
    case 'multiply': {
      const N = Math.max(6, Math.round(8 + t.handsUpLevel * 8 + t.movementLevel * 4))
      const r = baseR * 0.85 * entryScale * volMul
      for (let i = 0; i < N; i++) {
        const seed = t.id * 17 + i * 31
        const a = (seed % 100) / 100 * Math.PI * 2 + ts / 4000
        const dist = anchor.headSize * (0.6 + ((seed * 7) % 100) / 100 * 0.9 + Math.sin(ts / 1500 + i) * 0.1)
        const px = cx + Math.cos(a) * dist
        const py = cy + Math.sin(a) * dist * 0.8
        drawColoredLogo(ctx, px, py, r, Math.sin(ts / 900 + i) * 0.4, groupHue + skipColorHueShift)
      }
      break
    }
    case 'breathe': {
      const breath = 1 + Math.sin(ts / 1800) * 0.25
      const r = baseR * 1.5 * breath * entryScale * volMul
      drawColoredLogo(ctx, cx, cy, r, entryRotation, groupHue + skipColorHueShift)
      // 부드러운 후광
      ctx.save()
      ctx.globalCompositeOperation = 'screen'
      ctx.globalAlpha *= 0.3
      const grad = ctx.createRadialGradient(cx, cy, 0, cx, cy, r * 2.5)
      grad.addColorStop(0, hueShifted('rgba(29,185,84,0.7)', groupHue + skipColorHueShift))
      grad.addColorStop(1, 'rgba(29,185,84,0)')
      ctx.fillStyle = grad
      ctx.beginPath(); ctx.arc(cx, cy, r * 2.5, 0, Math.PI * 2); ctx.fill()
      ctx.restore()
      break
    }
    case 'pulse': {
      // 중앙 로고
      const r = baseR * entryScale * volMul
      drawColoredLogo(ctx, cx, cy, r, entryRotation, groupHue + skipColorHueShift)
      // 동심원 음파 (3겹, phase 다름)
      const PULSE_PERIOD = 1500
      ctx.save()
      ctx.lineWidth = 3
      for (let i = 0; i < 3; i++) {
        const phase = (((ts + (i * PULSE_PERIOD) / 3) % PULSE_PERIOD) / PULSE_PERIOD)
        const ringR = phase * anchor.headSize * (1.6 + t.movementLevel * 0.6)
        const a = 1 - phase
        ctx.globalAlpha = a * t.effectVisibility * entryAlpha * 0.7
        ctx.strokeStyle = hueShifted(SPOTIFY_GREEN, groupHue + skipColorHueShift)
        ctx.beginPath(); ctx.arc(cx, cy, ringR, 0, Math.PI * 2); ctx.stroke()
      }
      ctx.restore()
      break
    }
  }

  ctx.restore()

  drawExtras(ctx, t, anchor, vw, ts, mirrored, mode)
}

function drawExtras(
  ctx: CanvasRenderingContext2D,
  t: Track,
  anchor: ReturnType<typeof effectAnchor>,
  vw: number,
  ts: number,
  mirrored: boolean,
  mode: InteractionMode,
) {
  let cx = anchor.cx + t.togetherOffset.x
  if (mirrored) cx = vw - cx
  const cy = anchor.cy + t.togetherOffset.y

  // tap-like — 하트 + "+1" 위로 떠오름
  if (mode === 'tap-like' && t.likeBurstAt !== null) {
    const lt = (ts - t.likeBurstAt) / LIKE_BURST_MS
    const yRise = -lt * anchor.headSize * 0.6
    const alpha = 1 - lt
    ctx.save()
    ctx.globalAlpha *= alpha
    drawHeart(ctx, cx, cy + yRise, anchor.headSize * 0.12)
    ctx.font = 'bold 16px ui-monospace, monospace'
    ctx.fillStyle = '#ff5e9c'
    ctx.textAlign = 'center'
    ctx.fillText('+1', cx + anchor.headSize * 0.18, cy + yRise - anchor.headSize * 0.05)
    ctx.restore()
  }

  // drop-beat — 폭발 확산 로고들
  if (mode === 'drop-beat' && t.dropBurstAt !== null) {
    const lt = (ts - t.dropBurstAt) / DROP_BURST_MS
    const r = baseDropRadius(anchor.headSize, lt)
    const alpha = 1 - lt
    ctx.save()
    ctx.globalAlpha *= alpha
    const N = 10
    for (let i = 0; i < N; i++) {
      const a = (i / N) * Math.PI * 2
      const px = cx + Math.cos(a) * r
      const py = cy + Math.sin(a) * r
      drawSpotifyLogo(ctx, px, py, anchor.headSize * 0.08 * (1 - lt * 0.5), 0)
    }
    ctx.restore()
  }

  // discover — 손가락 가리킨 곳에 로고들
  if (mode === 'discover') {
    for (const d of t.discoverLogos) {
      const lt = (ts - d.bornAt) / d.ttl
      const alpha = lt < 0.2 ? (lt / 0.2) : (1 - (lt - 0.2) / 0.8)
      ctx.save()
      ctx.globalAlpha *= clamp(alpha, 0, 1) * t.effectVisibility
      const dx = mirrored ? vw - d.x : d.x
      drawSpotifyLogo(ctx, dx, d.y, anchor.headSize * 0.08, 0)
      ctx.restore()
    }
  }
}

function baseDropRadius(headSize: number, t: number) {
  return easeOutCubic(t) * headSize * 1.5
}

function drawHeart(ctx: CanvasRenderingContext2D, cx: number, cy: number, r: number) {
  ctx.save()
  ctx.translate(cx, cy)
  ctx.scale(r / 12, r / 12)
  ctx.fillStyle = '#ff5e9c'
  ctx.beginPath()
  ctx.moveTo(0, 4)
  ctx.bezierCurveTo(0, -2, -8, -6, -8, 0)
  ctx.bezierCurveTo(-8, 4, 0, 9, 0, 12)
  ctx.bezierCurveTo(0, 9, 8, 4, 8, 0)
  ctx.bezierCurveTo(8, -6, 0, -2, 0, 4)
  ctx.fill()
  ctx.restore()
}

function drawColoredLogo(ctx: CanvasRenderingContext2D, cx: number, cy: number, r: number, rotation: number, hueShift: number) {
  if (r < 0.5) return
  ctx.save()
  if (hueShift !== 0) {
    ctx.filter = `hue-rotate(${hueShift}deg)`
  }
  drawSpotifyLogo(ctx, cx, cy, r, rotation)
  ctx.restore()
}

function hueShifted(color: string, hueShift: number): string {
  if (hueShift === 0) return color
  // 단순 hsl 변환: hex/RGB → HSL → hueShift → RGB
  // 여기선 간단히 hsl shift된 spotify color 근사
  const baseHue = 141  // Spotify green base
  const newHue = (baseHue + hueShift) % 360
  return `hsl(${newHue}, 73%, 42%)`
}

function drawSpotifyLogo(ctx: CanvasRenderingContext2D, cx: number, cy: number, r: number, rotation = 0) {
  if (r < 0.5) return
  if (!spotifyImage.complete || spotifyImage.naturalWidth === 0) return
  ctx.save()
  ctx.translate(cx, cy)
  if (rotation) ctx.rotate(rotation)

  // Layer 1: 녹색 base 디스크 — 외곽 drop shadow 단일 솔리드 도형에만 적용 → 깔끔한 그림자
  ctx.shadowColor = 'rgba(0,0,0,0.35)'
  ctx.shadowBlur = r * 0.3
  ctx.shadowOffsetY = r * 0.08
  ctx.fillStyle = SPOTIFY_GREEN
  ctx.beginPath()
  ctx.arc(0, 0, r, 0, Math.PI * 2)
  ctx.fill()
  ctx.shadowColor = 'transparent'
  ctx.shadowBlur = 0
  ctx.shadowOffsetY = 0

  // Layer 2: 흰 내부 (SVG cutout 막대 채움). 외곽보다 충분히 작아 가장자리에 흰 띠 생기지 않음
  ctx.fillStyle = '#fff'
  ctx.beginPath()
  ctx.arc(0, 0, r * 0.85, 0, Math.PI * 2)
  ctx.fill()

  // Layer 3: 공식 SVG 오버레이 (외곽 녹색 + 막대 cutout)
  ctx.drawImage(spotifyImage, -r, -r, r * 2, r * 2)

  ctx.restore()
}

// ─── 시각화 ─────────────────────────────────────────────

function drawBBox(ctx: CanvasRenderingContext2D, t: Track, vw: number, mirrored: boolean) {
  let x = t.bbox.x
  if (mirrored) x = vw - t.bbox.x - t.bbox.w
  const { y, w, h } = t.bbox
  ctx.save()
  ctx.strokeStyle = colorForId(t.id)
  ctx.lineWidth = 3
  ctx.strokeRect(x, y, w, h)
  ctx.restore()
}

function drawHeadLabel(ctx: CanvasRenderingContext2D, t: Track, displayNum: number, vw: number, mirrored: boolean) {
  let cx = t.bbox.x + t.bbox.w / 2
  if (mirrored) cx = vw - cx
  const yTop = Math.max(28, t.bbox.y)
  const color = colorForId(t.id)
  const label = `#${displayNum}  ${(t.score * 100).toFixed(0)}%`
  ctx.save()
  ctx.font = 'bold 18px ui-monospace, Menlo, monospace'
  ctx.textAlign = 'center'
  ctx.textBaseline = 'bottom'
  const textW = ctx.measureText(label).width
  const padX = 10
  const labelH = 26
  const bx = cx - textW / 2 - padX
  const by = yTop - labelH - 6
  ctx.fillStyle = 'rgba(0, 0, 0, 0.78)'
  ctx.fillRect(bx, by, textW + padX * 2, labelH)
  ctx.strokeStyle = color
  ctx.lineWidth = 1.5
  ctx.strokeRect(bx, by, textW + padX * 2, labelH)
  ctx.fillStyle = color
  ctx.fillText(label, cx, by + labelH - 4)
  ctx.restore()
}

function drawSilhouette(
  ctx: CanvasRenderingContext2D,
  seg: ImageSegmenterResult,
  vw: number,
  vh: number,
  mirrored: boolean,
  mode: ShapeMode,
  tracks: Track[],
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

  // selfie_multiclass — 0=background, 1=hair, 2=body-skin, 3=face-skin, 4=clothes, 5=others/accessories
  // 0이 아니면 사람의 일부로 간주
  const rawIsPerson = (v: number) => v !== 0

  // 트랙 bbox 안쪽 픽셀만 통과 — bbox 밖에 있는 가구/배경 false-positive 차단
  const sx = vw / w
  const sy = vh / h
  // 패딩 축소: 멀티클래스 모델이 더 정확하므로 작은 마진만 (10% 또는 12px)
  const padded = tracks.map((t) => {
    const pad = Math.max(t.bbox.w * 0.10, 12)
    return {
      x1: t.bbox.x - pad,
      y1: t.bbox.y - pad,
      x2: t.bbox.x + t.bbox.w + pad,
      y2: t.bbox.y + t.bbox.h + pad,
    }
  })
  const hasTracks = padded.length > 0
  const isPersonFn = (v: number, xx: number, yy: number) => {
    if (!rawIsPerson(v)) return false
    if (!hasTracks) return false  // 트랙 없으면 전부 background 처리 (가구 false-positive 제거)
    const px = xx * sx, py = yy * sy
    for (const b of padded) {
      if (px >= b.x1 && px <= b.x2 && py >= b.y1 && py <= b.y2) return true
    }
    return false
  }

  // 1) raw person mask (multiclass + bbox)
  const rawMask = new Uint8Array(data.length)
  for (let yy = 0; yy < h; yy++) {
    for (let xx = 0; xx < w; xx++) {
      const i = yy * w + xx
      if (isPersonFn(data[i], xx, yy)) rawMask[i] = 1
    }
  }

  // 2) 얇은 연결(예: 손→베개) 끊기 위해 erode
  const eroded = erodeBinary(rawMask, w, h, 2)

  // 3) 각 트랙의 얼굴/상체 부근 seed에서만 flood-fill → 본체와 연결된 영역만 유지
  //    (베개/의자는 erode로 본체와 분리되므로 flood-fill에 안 잡힘)
  const seeds: number[] = []
  for (const t of tracks) {
    const cxv = t.bbox.x + t.bbox.w * 0.5
    // 얼굴 가능성 높은 위치 여러 곳을 후보로
    const ys = [t.bbox.y + t.bbox.h * 0.18, t.bbox.y + t.bbox.h * 0.32, t.bbox.y + t.bbox.h * 0.5]
    for (const yv of ys) {
      const mx = Math.floor(cxv / sx)
      const my = Math.floor(yv / sy)
      if (mx >= 0 && mx < w && my >= 0 && my < h) seeds.push(my * w + mx)
    }
  }
  const flooded = floodFillFromSeeds(eroded, w, h, seeds)

  // 4) dilate 3 — erode로 줄어든 크기 복구
  const personMask = dilateBinary(flooded, w, h, 3)

  if (mode === 'silhouette-outline') {
    // 윤곽선 — 두꺼운 stroke 만들기 위해 반경 3까지의 이웃 비교(dilation 효과)
    const baseR = 255, baseG = 60, baseB = 60
    const radius = 3
    for (let yy = 0; yy < h; yy++) {
      for (let xx = 0; xx < w; xx++) {
        const i = yy * w + xx
        const p = personMask[i]
        let isEdge = false
        for (let dy = -radius; dy <= radius && !isEdge; dy++) {
          const ny = yy + dy
          if (ny < 0 || ny >= h) continue
          for (let dx = -radius; dx <= radius && !isEdge; dx++) {
            if (dx === 0 && dy === 0) continue
            const nx = xx + dx
            if (nx < 0 || nx >= w) continue
            if (personMask[ny * w + nx] !== p) isEdge = true
          }
        }
        const o = i * 4
        if (isEdge) {
          img.data[o] = baseR; img.data[o + 1] = baseG; img.data[o + 2] = baseB; img.data[o + 3] = 230
        } else {
          img.data[o] = 0; img.data[o + 1] = 0; img.data[o + 2] = 0; img.data[o + 3] = 0
        }
      }
    }
  } else {
    const targetIsPerson = mode === 'silhouette-fg'
    const baseR = targetIsPerson ? 255 : 0
    const baseG = targetIsPerson ? 120 : 255
    const baseB = targetIsPerson ? 60 : 200
    for (let i = 0; i < data.length; i++) {
      const matches = targetIsPerson ? (personMask[i] === 1) : (personMask[i] === 0)
      const o = i * 4
      if (matches) {
        img.data[o] = baseR; img.data[o + 1] = baseG; img.data[o + 2] = baseB; img.data[o + 3] = 120
      } else {
        img.data[o] = 0; img.data[o + 1] = 0; img.data[o + 2] = 0; img.data[o + 3] = 0
      }
    }
  }
  offCtx.putImageData(img, 0, 0)

  ctx.save()
  if (mirrored) { ctx.translate(vw, 0); ctx.scale(-1, 1) }
  if (mode === 'silhouette-outline') {
    // 1) 부드러운 외곽 글로우 (큰 블러)
    ctx.imageSmoothingEnabled = true
    ctx.imageSmoothingQuality = 'high'
    ctx.shadowColor = 'transparent'
    ctx.filter = 'blur(10px)'
    ctx.globalCompositeOperation = 'lighter'
    ctx.drawImage(off, 0, 0, vw, vh)
    // 2) 또렷한 안쪽 outline (작은 블러 — 지글거림 제거)
    ctx.filter = 'blur(2.5px)'
    ctx.drawImage(off, 0, 0, vw, vh)
    ctx.filter = 'none'
    ctx.globalCompositeOperation = 'source-over'
  } else {
    ctx.globalCompositeOperation = 'screen'
    ctx.drawImage(off, 0, 0, vw, vh)
    if (mode === 'silhouette-fg') {
      ctx.globalCompositeOperation = 'source-over'
      ctx.shadowColor = 'rgba(255, 120, 60, 0.9)'
      ctx.shadowBlur = 16
      ctx.drawImage(off, 0, 0, vw, vh)
      ctx.shadowBlur = 0
    }
  }
  ctx.restore()
}

const COLORS = ['#7ee', '#ff7', '#f7f', '#7f7', '#f77', '#77f', '#fa7', '#7fa', '#a7f', '#f7a']
function colorForId(id: number): string { return COLORS[id % COLORS.length] }

// ─── 마스크 morphology / flood-fill ────────────────────

function erodeBinary(src: Uint8Array, w: number, h: number, steps: number): Uint8Array {
  let curr = src
  for (let s = 0; s < steps; s++) {
    const next = new Uint8Array(curr.length)
    for (let yy = 1; yy < h - 1; yy++) {
      const row = yy * w
      for (let xx = 1; xx < w - 1; xx++) {
        const i = row + xx
        if (curr[i] && curr[i - 1] && curr[i + 1] && curr[i - w] && curr[i + w]) next[i] = 1
      }
    }
    curr = next
  }
  return curr
}

function dilateBinary(src: Uint8Array, w: number, h: number, steps: number): Uint8Array {
  let curr = src
  for (let s = 0; s < steps; s++) {
    const next = new Uint8Array(curr.length)
    for (let yy = 0; yy < h; yy++) {
      const row = yy * w
      for (let xx = 0; xx < w; xx++) {
        const i = row + xx
        if (curr[i]) { next[i] = 1; continue }
        if (xx > 0 && curr[i - 1]) { next[i] = 1; continue }
        if (xx < w - 1 && curr[i + 1]) { next[i] = 1; continue }
        if (yy > 0 && curr[i - w]) { next[i] = 1; continue }
        if (yy < h - 1 && curr[i + w]) { next[i] = 1; continue }
      }
    }
    curr = next
  }
  return curr
}

function floodFillFromSeeds(mask: Uint8Array, w: number, h: number, seedIndices: number[]): Uint8Array {
  const out = new Uint8Array(mask.length)
  const visited = new Uint8Array(mask.length)
  const stack: number[] = []
  for (const seed of seedIndices) {
    if (seed < 0 || seed >= mask.length) continue
    if (!mask[seed] || visited[seed]) continue
    stack.push(seed)
    while (stack.length > 0) {
      const j = stack.pop()!
      if (visited[j] || !mask[j]) continue
      visited[j] = 1
      out[j] = 1
      const xx = j % w
      const yy = (j - xx) / w
      if (xx > 0) stack.push(j - 1)
      if (xx < w - 1) stack.push(j + 1)
      if (yy > 0) stack.push(j - w)
      if (yy < h - 1) stack.push(j + w)
    }
  }
  return out
}

// ─── UI ──────────────────────────────────────────────────

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
          <Row label="Display">
            {(['none', 'box', 'silhouette-bg', 'silhouette-fg', 'silhouette-outline'] as ShapeMode[]).map((s) => (
              <Toggle key={s} on={shape === s} onClick={() => setShape(s)}>{shapeLabel(s)}</Toggle>
            ))}
          </Row>
          <Row label="Effect">
            {(['none', 'pop', 'bounce', 'orbit', 'multiply', 'breathe', 'pulse'] as EffectType[]).map((e) => (
              <Toggle key={e} on={effect === e} onClick={() => setEffect(e)}>{effectLabel(e)}</Toggle>
            ))}
          </Row>
          <Row label="Interaction">
            {(['none', 'move-music', 'volume-up', 'tap-like', 'listen-together', 'drop-beat', 'skip-track', 'headphones', 'discover', 'group-sync'] as InteractionMode[]).map((m) => (
              <Toggle key={m} on={interaction === m} onClick={() => setInteraction(m)}>{interactionLabel(m)}</Toggle>
            ))}
          </Row>
        </div>
      )}
      <button type="button" onClick={toggle} title="Effects panel" style={fabStyle(open)}>
        <SparkleIcon />
      </button>
    </div>
  )
}

function shapeLabel(s: ShapeMode): string {
  return ({ 'none': 'None', 'box': 'Box', 'silhouette-bg': 'Silhouette-bg', 'silhouette-fg': 'Silhouette-fg', 'silhouette-outline': 'Outline' } as const)[s]
}
function effectLabel(e: EffectType): string {
  return ({ 'none': 'None', 'pop': 'Pop', 'bounce': 'Bounce', 'orbit': 'Orbit', 'multiply': 'Multiply', 'breathe': 'Breathe', 'pulse': 'Pulse' } as const)[e]
}
function interactionLabel(m: InteractionMode): string {
  return ({
    'none': 'None',
    'move-music': 'Move',
    'volume-up': 'Volume',
    'tap-like': 'Like',
    'listen-together': 'Together',
    'drop-beat': 'Drop',
    'skip-track': 'Skip',
    'headphones': 'Headphones',
    'discover': 'Discover',
    'group-sync': 'Sync',
  } as const)[m]
}

function FullscreenButton({ isFullscreen }: { isFullscreen: boolean }) {
  function onClick() {
    if (isFullscreen) document.exitFullscreen().catch(() => {})
    else document.documentElement.requestFullscreen().catch(() => {})
  }
  return (
    <button type="button" onClick={onClick} style={fsBtnStyle} title={isFullscreen ? 'Exit fullscreen' : 'Enter fullscreen'}>
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
      <path d="M4 9 V4 H9" /><path d="M20 9 V4 H15" /><path d="M4 15 V20 H9" /><path d="M20 15 V20 H15" />
    </svg>
  )
}
function FullscreenExitIcon() {
  return (
    <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.2" strokeLinecap="round" strokeLinejoin="round" aria-hidden>
      <path d="M9 4 V9 H4" /><path d="M15 4 V9 H20" /><path d="M9 20 V15 H4" /><path d="M15 20 V15 H20" />
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
  position: 'fixed', top: 0, left: 0, width: 28, height: 28,
  background: 'transparent', border: 'none', cursor: 'pointer', padding: 0, zIndex: 11,
}
const statusOverlayStyle: React.CSSProperties = {
  position: 'fixed', top: 14, left: 14, zIndex: 10,
  display: 'flex', alignItems: 'center', gap: 8,
  fontSize: 13, fontWeight: 600,
  fontFamily: 'ui-monospace, SFMono-Regular, Menlo, monospace',
  color: '#fff', textShadow: '0 1px 3px rgba(0,0,0,0.85)',
  pointerEvents: 'none', maxWidth: 'calc(100vw - 28px)',
}
const bottomWrapStyle: React.CSSProperties = {
  position: 'fixed', left: '50%', bottom: 16, transform: 'translateX(-50%)',
  zIndex: 10, display: 'flex', flexDirection: 'column', alignItems: 'center', gap: 10,
  maxWidth: 'calc(100vw - 28px)', width: 'max-content',
  fontFamily: 'ui-monospace, SFMono-Regular, Menlo, monospace',
  color: '#fff', textShadow: '0 1px 2px rgba(0,0,0,0.85)',
}
const panelStyle: React.CSSProperties = {
  display: 'flex', flexDirection: 'column', gap: 8,
  fontSize: 13, maxWidth: 'min(720px, calc(100vw - 28px))', alignItems: 'flex-start',
}
const rowStyle: React.CSSProperties = {
  display: 'flex', alignItems: 'center', gap: 10, flexWrap: 'wrap',
}
const rowLabelStyle: React.CSSProperties = {
  fontSize: 11, opacity: 0.7, letterSpacing: 0.5, textTransform: 'uppercase', minWidth: 70,
}
const rowChildrenStyle: React.CSSProperties = {
  display: 'flex', flexWrap: 'wrap', gap: 6,
}
const toggleStyle = (on: boolean): React.CSSProperties => ({
  background: on ? 'rgba(29,185,84,0.22)' : 'rgba(255,255,255,0.04)',
  border: `1px solid ${on ? 'rgba(29,185,84,0.8)' : 'rgba(255,255,255,0.25)'}`,
  color: '#fff', padding: '4px 10px', borderRadius: 6, cursor: 'pointer',
  fontSize: 12, fontFamily: 'inherit', textShadow: '0 1px 2px rgba(0,0,0,0.6)',
})
const fabStyle = (open: boolean): React.CSSProperties => ({
  width: 48, height: 48, borderRadius: '50%',
  border: `1px solid ${open ? 'rgba(29,185,84,0.7)' : 'rgba(255,255,255,0.4)'}`,
  background: open ? 'rgba(29,185,84,0.18)' : 'rgba(0,0,0,0.55)',
  backdropFilter: 'blur(8px)',
  color: '#fff', cursor: 'pointer', padding: 0,
  display: 'flex', alignItems: 'center', justifyContent: 'center',
  boxShadow: open ? '0 0 16px rgba(29,185,84,0.5)' : '0 2px 8px rgba(0,0,0,0.5)',
})
const fsBtnStyle: React.CSSProperties = {
  position: 'fixed', right: 16, bottom: 16, zIndex: 11,
  width: 42, height: 42, borderRadius: '50%',
  border: '1px solid rgba(255,255,255,0.4)',
  background: 'rgba(0,0,0,0.55)', backdropFilter: 'blur(8px)',
  color: '#fff', cursor: 'pointer', padding: 0,
  display: 'flex', alignItems: 'center', justifyContent: 'center',
  boxShadow: '0 2px 8px rgba(0,0,0,0.5)', transition: 'opacity 200ms',
}

// 사용 안되는 const 경고 회피
void HEAD_DIST_NEAR_X
