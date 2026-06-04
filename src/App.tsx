import { useEffect, useRef, useState } from 'react'
import { FilesetResolver, ObjectDetector, type ObjectDetectorResult } from '@mediapipe/tasks-vision'

const WASM_URL = 'https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.35/wasm'
const MODEL_URL = 'https://storage.googleapis.com/mediapipe-models/object_detector/efficientdet_lite0/float32/1/efficientdet_lite0.tflite'

const MAX_DETECTIONS = 20            // 모델에 요청할 최대 detection 개수
const TRACK_TIMEOUT_MS = 800         // 사라진 트랙을 유지할 시간
const TRACK_MATCH_IOU = 0.2          // 같은 사람으로 매칭할 IoU 임계
const SCORE_THRESHOLD = 0.45
const POSITION_ALPHA = 0.35          // bbox 위치/크기 EMA — 낮을수록 부드러움 (응답 ↓)
const SCORE_ALPHA = 0.15             // 인식률 EMA — 더 느리게 (텍스트 떨림 방지)

type Status = 'idle' | 'loading-model' | 'requesting-camera' | 'running' | 'error'

type BBox = { x: number; y: number; w: number; h: number }

type Track = {
  id: number
  bbox: BBox
  score: number
  firstSeenAt: number
  lastSeenAt: number
}

export default function App() {
  const videoRef = useRef<HTMLVideoElement>(null)
  const canvasRef = useRef<HTMLCanvasElement>(null)
  const detectorRef = useRef<ObjectDetector | null>(null)

  const [status, setStatus] = useState<Status>('idle')
  const [errorMsg, setErrorMsg] = useState('')
  const [mirror, setMirror] = useState(true)
  const [showOverlay, setShowOverlay] = useState(true)
  const [debug, setDebug] = useState(false)
  const [fps, setFps] = useState(0)
  const [activeTrackCount, setActiveTrackCount] = useState(0)

  const mirrorRef = useRef(mirror)
  const showOverlayRef = useRef(showOverlay)
  const debugRef = useRef(debug)
  useEffect(() => { mirrorRef.current = mirror }, [mirror])
  useEffect(() => { showOverlayRef.current = showOverlay }, [showOverlay])
  useEffect(() => { debugRef.current = debug }, [debug])

  const tracksRef = useRef<Track[]>([])
  const nextIdRef = useRef(1)

  useEffect(() => {
    function onKey(e: KeyboardEvent) {
      if (e.target instanceof HTMLInputElement || e.target instanceof HTMLTextAreaElement) return
      const k = e.key.toLowerCase()
      if (k === 'd') setDebug((v) => !v)
      else if (k === 'm') setMirror((v) => !v)
      else if (k === 'o') setShowOverlay((v) => !v)
      else if (k === 'f') toggleFullscreen()
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
        const detector = await ObjectDetector.createFromOptions(vision, {
          baseOptions: { modelAssetPath: MODEL_URL, delegate: 'GPU' },
          runningMode: 'VIDEO',
          scoreThreshold: SCORE_THRESHOLD,
          maxResults: MAX_DETECTIONS,
          categoryAllowlist: ['person'],
        })
        if (cancelled) {
          detector.close()
          return
        }
        detectorRef.current = detector

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
      detectorRef.current?.close()
      detectorRef.current = null
    }
  }, [])

  function detectAndRender(ts: number) {
    const video = videoRef.current
    const canvas = canvasRef.current
    const detector = detectorRef.current
    if (!video || !canvas || !detector) return
    if (video.readyState < 2) return

    const vw = video.videoWidth
    const vh = video.videoHeight
    if (canvas.width !== vw) canvas.width = vw
    if (canvas.height !== vh) canvas.height = vh

    const ctx = canvas.getContext('2d')!
    const mirrored = mirrorRef.current

    ctx.save()
    if (mirrored) {
      ctx.translate(vw, 0)
      ctx.scale(-1, 1)
    }
    ctx.drawImage(video, 0, 0, vw, vh)
    ctx.restore()

    let result: ObjectDetectorResult | undefined
    try {
      result = detector.detectForVideo(video, ts)
    } catch {
      return
    }

    const detections: { bbox: BBox; score: number }[] = []
    if (result) {
      for (const d of result.detections) {
        const c = d.categories?.[0]
        if (!c || c.categoryName !== 'person') continue
        if (c.score < SCORE_THRESHOLD) continue
        const b = d.boundingBox
        if (!b) continue
        detections.push({
          bbox: { x: b.originX, y: b.originY, w: b.width, h: b.height },
          score: c.score,
        })
      }
    }

    updateTracks(tracksRef.current, detections, ts, nextIdRef)
    if (tracksRef.current.length !== activeTrackCount) {
      setActiveTrackCount(tracksRef.current.length)
    }

    if (showOverlayRef.current) {
      for (const t of tracksRef.current) {
        drawAuraForBBox(ctx, t.bbox, vw, mirrored)
      }
    }

    if (debugRef.current) {
      for (const t of tracksRef.current) {
        drawDebugBox(ctx, t, vw, ts, mirrored)
      }
    }
  }

  return (
    <div style={containerStyle}>
      <video ref={videoRef} playsInline muted style={{ display: 'none' }} />
      <canvas ref={canvasRef} style={canvasStyle} />

      {debug && (
        <DebugBar
          status={status}
          errorMsg={errorMsg}
          fps={fps}
          trackCount={activeTrackCount}
          mirror={mirror}
          showOverlay={showOverlay}
          onToggleMirror={() => setMirror((m) => !m)}
          onToggleOverlay={() => setShowOverlay((s) => !s)}
          onToggleFullscreen={toggleFullscreen}
        />
      )}

      <button
        type="button"
        title="디버그 토글 (D)"
        onClick={() => setDebug((d) => !d)}
        style={debugToggleStyle(debug)}
      />
    </div>
  )
}

// ─── 트래커 ───────────────────────────────────────────────

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
      if (iou > bestIoU) {
        bestIoU = iou
        bestIdx = i
      }
    }
    if (bestIdx >= 0) {
      const det = detections[bestIdx]
      // EMA 스무딩: bbox 위치/크기 떨림 완화
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
    tracks.push({
      id: nextIdRef.current++,
      bbox: detections[i].bbox,
      score: detections[i].score,
      firstSeenAt: now,
      lastSeenAt: now,
    })
  }

  for (let i = tracks.length - 1; i >= 0; i--) {
    if (now - tracks[i].lastSeenAt > TRACK_TIMEOUT_MS) {
      tracks.splice(i, 1)
    }
  }
}

function lerp(a: number, b: number, t: number): number {
  return a + (b - a) * t
}

function iouOf(a: BBox, b: BBox): number {
  const x1 = Math.max(a.x, b.x)
  const y1 = Math.max(a.y, b.y)
  const x2 = Math.min(a.x + a.w, b.x + b.w)
  const y2 = Math.min(a.y + a.h, b.y + b.h)
  if (x2 <= x1 || y2 <= y1) return 0
  const inter = (x2 - x1) * (y2 - y1)
  const union = a.w * a.h + b.w * b.h - inter
  return union > 0 ? inter / union : 0
}

// ─── 렌더 ─────────────────────────────────────────────────

function drawAuraForBBox(ctx: CanvasRenderingContext2D, bbox: BBox, vw: number, mirrored: boolean) {
  const headSize = Math.max(40, bbox.w * 0.35)
  let cx = bbox.x + bbox.w / 2
  const cy = bbox.y + headSize * 0.5
  if (mirrored) cx = vw - cx

  const r = headSize * 1.1
  const grad = ctx.createRadialGradient(cx, cy, 0, cx, cy, r)
  grad.addColorStop(0, 'rgba(255, 230, 130, 0.95)')
  grad.addColorStop(0.45, 'rgba(255, 170, 60, 0.5)')
  grad.addColorStop(1, 'rgba(255, 80, 0, 0)')

  ctx.save()
  ctx.globalCompositeOperation = 'screen'
  ctx.fillStyle = grad
  ctx.beginPath()
  ctx.arc(cx, cy, r, 0, Math.PI * 2)
  ctx.fill()
  ctx.restore()
}

function drawDebugBox(ctx: CanvasRenderingContext2D, t: Track, vw: number, now: number, mirrored: boolean) {
  let x = t.bbox.x
  if (mirrored) x = vw - t.bbox.x - t.bbox.w
  const { y, w, h } = t.bbox
  const dwellSec = (now - t.firstSeenAt) / 1000
  const dwellStr = dwellSec < 10 ? dwellSec.toFixed(1) : Math.round(dwellSec).toString()
  const color = colorForId(t.id)

  ctx.save()

  // 박스 (테두리)
  ctx.strokeStyle = color
  ctx.lineWidth = 3
  ctx.strokeRect(x, y, w, h)

  // 라벨
  const label = `#${t.id}   ${dwellStr}s   ${(t.score * 100).toFixed(0)}%`
  ctx.font = 'bold 18px ui-monospace, Menlo, monospace'
  const textW = ctx.measureText(label).width
  const padX = 10
  const padY = 6
  const labelH = 28
  const labelY = Math.max(0, y - labelH - 2)

  // 라벨 배경 (반투명 검정 + 컬러 외곽)
  ctx.fillStyle = 'rgba(0, 0, 0, 0.85)'
  ctx.fillRect(x, labelY, textW + padX * 2, labelH)
  ctx.strokeStyle = color
  ctx.lineWidth = 2
  ctx.strokeRect(x, labelY, textW + padX * 2, labelH)

  // 텍스트
  ctx.fillStyle = color
  ctx.textBaseline = 'middle'
  ctx.fillText(label, x + padX, labelY + labelH / 2 + 1)

  ctx.restore()
}

const COLORS = ['#7ee', '#ff7', '#f7f', '#7f7', '#f77', '#77f', '#fa7', '#7fa', '#a7f', '#f7a']
function colorForId(id: number): string {
  return COLORS[id % COLORS.length]
}

// ─── 컴포넌트 ─────────────────────────────────────────────

function DebugBar(props: {
  status: Status
  errorMsg: string
  fps: number
  trackCount: number
  mirror: boolean
  showOverlay: boolean
  onToggleMirror: () => void
  onToggleOverlay: () => void
  onToggleFullscreen: () => void
}) {
  const { status, errorMsg, fps, trackCount, mirror, showOverlay, onToggleMirror, onToggleOverlay, onToggleFullscreen } = props

  const statusLabel: Record<Status, string> = {
    idle: '⚪ idle',
    'loading-model': '⏳ 모델 로딩…',
    'requesting-camera': '📷 카메라 권한 요청 중…',
    running: '🟢 running',
    error: '🔴 error',
  }

  const fpsColor = fps >= 50 ? '#7ee' : fps >= 30 ? '#ff7' : '#f77'

  return (
    <div style={debugBarStyle}>
      <span>{statusLabel[status]}</span>
      {status === 'running' && (
        <>
          <span style={{ opacity: 0.5 }}>·</span>
          <span style={{ color: fpsColor }}>{fps} fps</span>
          <span style={{ opacity: 0.5 }}>·</span>
          <span>👥 {trackCount}명</span>
        </>
      )}
      {status === 'error' && <span style={{ color: '#f77' }}>· {errorMsg}</span>}
      <span style={{ flex: 1 }} />
      <button onClick={onToggleMirror} style={btnStyle}>거울 {mirror ? '●' : '○'} <kbd style={kbdStyle}>M</kbd></button>
      <button onClick={onToggleOverlay} style={btnStyle}>효과 {showOverlay ? '●' : '○'} <kbd style={kbdStyle}>O</kbd></button>
      <button onClick={onToggleFullscreen} style={btnStyle}>풀스크린 <kbd style={kbdStyle}>F</kbd></button>
    </div>
  )
}

function toggleFullscreen() {
  if (!document.fullscreenElement) {
    document.documentElement.requestFullscreen().catch(() => {})
  } else {
    document.exitFullscreen().catch(() => {})
  }
}

// ─── 스타일 ───────────────────────────────────────────────

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

const debugBarStyle: React.CSSProperties = {
  position: 'fixed',
  top: 16,
  left: 16,
  right: 16,
  display: 'flex',
  alignItems: 'center',
  gap: 10,
  padding: '10px 14px',
  background: 'rgba(0,0,0,0.6)',
  backdropFilter: 'blur(8px)',
  border: '1px solid rgba(255,255,255,0.1)',
  borderRadius: 10,
  color: '#fff',
  fontSize: 13,
  fontFamily: 'ui-monospace, SFMono-Regular, Menlo, monospace',
  zIndex: 10,
}

const btnStyle: React.CSSProperties = {
  background: 'rgba(255,255,255,0.12)',
  border: '1px solid rgba(255,255,255,0.18)',
  color: '#fff',
  padding: '6px 10px',
  borderRadius: 6,
  cursor: 'pointer',
  fontSize: 12,
  fontFamily: 'inherit',
  display: 'inline-flex',
  alignItems: 'center',
  gap: 6,
}

const kbdStyle: React.CSSProperties = {
  background: 'rgba(0,0,0,0.4)',
  border: '1px solid rgba(255,255,255,0.18)',
  borderRadius: 3,
  padding: '0 4px',
  fontSize: 10,
  fontFamily: 'inherit',
}

const debugToggleStyle = (on: boolean): React.CSSProperties => ({
  position: 'fixed',
  right: 14,
  bottom: 14,
  width: 14,
  height: 14,
  borderRadius: '50%',
  border: '1px solid rgba(255,255,255,0.3)',
  background: on ? '#7ee' : 'rgba(255,255,255,0.15)',
  cursor: 'pointer',
  padding: 0,
  zIndex: 10,
  opacity: on ? 1 : 0.5,
})
