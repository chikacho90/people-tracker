import { useEffect, useRef, useState } from 'react'
import { FilesetResolver, PoseLandmarker, type PoseLandmarkerResult } from '@mediapipe/tasks-vision'

const WASM_URL = 'https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.35/wasm'
const MODEL_URL = 'https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_lite/float16/1/pose_landmarker_lite.task'

type Status = 'idle' | 'loading-model' | 'requesting-camera' | 'running' | 'error'

export default function App() {
  const videoRef = useRef<HTMLVideoElement>(null)
  const canvasRef = useRef<HTMLCanvasElement>(null)
  const poseRef = useRef<PoseLandmarker | null>(null)

  const [status, setStatus] = useState<Status>('idle')
  const [errorMsg, setErrorMsg] = useState('')
  const [mirror, setMirror] = useState(true)
  const [showOverlay, setShowOverlay] = useState(true)
  const [fps, setFps] = useState(0)

  // 최신 토글 값을 RAF 루프 안에서 읽기 위한 ref
  const mirrorRef = useRef(mirror)
  const showOverlayRef = useRef(showOverlay)
  useEffect(() => { mirrorRef.current = mirror }, [mirror])
  useEffect(() => { showOverlayRef.current = showOverlay }, [showOverlay])

  useEffect(() => {
    let cancelled = false
    let raf: number | null = null
    let lastTs = performance.now()
    const fpsBuf: number[] = []

    async function init() {
      try {
        setStatus('loading-model')
        const vision = await FilesetResolver.forVisionTasks(WASM_URL)
        const pose = await PoseLandmarker.createFromOptions(vision, {
          baseOptions: { modelAssetPath: MODEL_URL, delegate: 'GPU' },
          runningMode: 'VIDEO',
          numPoses: 1,
        })
        if (cancelled) {
          pose.close()
          return
        }
        poseRef.current = pose

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
      poseRef.current?.close()
      poseRef.current = null
    }
  }, [])

  function detectAndRender(ts: number) {
    const video = videoRef.current
    const canvas = canvasRef.current
    const pose = poseRef.current
    if (!video || !canvas || !pose) return
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

    let result: PoseLandmarkerResult | undefined
    try {
      result = pose.detectForVideo(video, ts)
    } catch {
      return
    }

    if (!result || result.landmarks.length === 0) return
    if (!showOverlayRef.current) return

    for (const landmarks of result.landmarks) {
      drawAura(ctx, landmarks, vw, vh, mirrored)
    }
  }

  return (
    <div style={containerStyle}>
      <video ref={videoRef} playsInline muted style={{ display: 'none' }} />
      <canvas ref={canvasRef} style={canvasStyle} />

      <ControlBar
        status={status}
        errorMsg={errorMsg}
        fps={fps}
        mirror={mirror}
        showOverlay={showOverlay}
        onToggleMirror={() => setMirror((m) => !m)}
        onToggleOverlay={() => setShowOverlay((s) => !s)}
      />
    </div>
  )
}

// 머리 위 후광 (placeholder) 그리기
function drawAura(
  ctx: CanvasRenderingContext2D,
  landmarks: Array<{ x: number; y: number; z: number; visibility?: number }>,
  vw: number,
  vh: number,
  mirrored: boolean,
) {
  const NOSE = 0
  const LEFT_SHOULDER = 11
  const RIGHT_SHOULDER = 12

  const nose = landmarks[NOSE]
  const ls = landmarks[LEFT_SHOULDER]
  const rs = landmarks[RIGHT_SHOULDER]
  if (!nose || !ls || !rs) return

  const shoulderW = Math.abs(ls.x - rs.x) * vw
  const headSize = Math.max(60, shoulderW * 0.6)

  let cx = nose.x * vw
  const cy = nose.y * vh - headSize * 0.8
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

function ControlBar(props: {
  status: Status
  errorMsg: string
  fps: number
  mirror: boolean
  showOverlay: boolean
  onToggleMirror: () => void
  onToggleOverlay: () => void
}) {
  const { status, errorMsg, fps, mirror, showOverlay, onToggleMirror, onToggleOverlay } = props

  const statusLabel: Record<Status, string> = {
    idle: '⚪ idle',
    'loading-model': '⏳ MediaPipe 모델 로딩…',
    'requesting-camera': '📷 카메라 권한 요청 중…',
    running: '🟢 running',
    error: '🔴 error',
  }

  const fpsColor = fps >= 50 ? '#7ee' : fps >= 30 ? '#ff7' : '#f77'

  return (
    <div style={controlBarStyle}>
      <strong>aura PoC</strong>
      <span style={{ opacity: 0.6 }}>·</span>
      <span>{statusLabel[status]}</span>
      {status === 'running' && (
        <>
          <span style={{ opacity: 0.6 }}>·</span>
          <span style={{ color: fpsColor }}>{fps} fps</span>
        </>
      )}
      {status === 'error' && (
        <>
          <span style={{ opacity: 0.6 }}>·</span>
          <span style={{ color: '#f77' }}>{errorMsg}</span>
        </>
      )}
      <span style={{ flex: 1 }} />
      <button onClick={onToggleMirror} style={btnStyle}>
        거울 {mirror ? 'ON' : 'OFF'}
      </button>
      <button onClick={onToggleOverlay} style={btnStyle}>
        효과 {showOverlay ? 'ON' : 'OFF'}
      </button>
      <button onClick={toggleFullscreen} style={btnStyle}>
        풀스크린
      </button>
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

const controlBarStyle: React.CSSProperties = {
  position: 'fixed',
  top: 16,
  left: 16,
  right: 16,
  display: 'flex',
  alignItems: 'center',
  gap: 10,
  padding: '10px 14px',
  background: 'rgba(0,0,0,0.55)',
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
}
