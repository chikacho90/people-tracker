# people-tracker

백화점/매장 인터랙티브 디스플레이를 위한 실시간 사람 인식 + 트래킹 PoC.
카메라 영상에서 사람을 검출하고 각 사람에 ID/체류시간을 부여, 머리 위에 시각 효과(후광 등) 합성.

거울 모드 (사람이 화면에서 자기 모습 보면서 인터랙션) + 풀스크린 키오스크 운영 가능.

## 기술
- React 19 + Vite + TypeScript
- MediaPipe Tasks Vision (Object Detector — `person` 클래스)
- Canvas 2D + WebGL 호환 합성
- 간단한 IoU 기반 자체 트래커 (외부 lib 없이 ID 유지)

## 운영
- Dev: `bun dev` → http://localhost:5173
- Prod 빌드: `bun run build`
- Vercel 자동 배포 (main 푸시 시): https://aura-rho-nine.vercel.app
- 디버그 모드: 화면 우하단 작은 버튼 또는 키보드 `D` 키

## 단축키
| 키 | 동작 |
|---|---|
| `D` | 디버그 토글 (FPS, bbox, ID, dwell time) |
| `M` | 거울 모드 토글 |
| `F` | 풀스크린 토글 |
| `O` | 효과(후광) ON/OFF |
