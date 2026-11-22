import { useEffect, useRef, useState } from 'react'
import {
  Accordion, AccordionDetails, AccordionSummary, Box, Button, Checkbox,
  Chip, Container, FormControl, FormControlLabel, InputLabel,
  LinearProgress, MenuItem, Select, Slider, Stack, TextField,
  Typography, Alert
} from '@mui/material'
import CameraAltIcon from '@mui/icons-material/CameraAlt'
import PlayArrowIcon from '@mui/icons-material/PlayArrow'
import PauseIcon from '@mui/icons-material/Pause'
import StopIcon from '@mui/icons-material/Stop'
import ReplayIcon from '@mui/icons-material/Replay'
import SwitchCameraIcon from '@mui/icons-material/Cameraswitch'
import ExpandMoreIcon from '@mui/icons-material/ExpandMore'
import './App.css'

const OCR_REQUEST_TIMEOUT_MS = 90_000

function makeCaptureCanvas(video, zoom = 1) {
  const canvas = document.createElement('canvas')
  const sourceWidth = video.videoWidth / zoom
  const sourceHeight = video.videoHeight / zoom
  const sourceX = (video.videoWidth - sourceWidth) / 2
  const sourceY = (video.videoHeight - sourceHeight) / 2
  const scale = Math.min(1600 / Math.max(sourceWidth, sourceHeight), 2)
  canvas.width = Math.round(sourceWidth * scale)
  canvas.height = Math.round(sourceHeight * scale)
  const ctx = canvas.getContext('2d')
  ctx.imageSmoothingEnabled = true
  ctx.imageSmoothingQuality = 'high'
  ctx.drawImage(
    video,
    sourceX,
    sourceY,
    sourceWidth,
    sourceHeight,
    0,
    0,
    canvas.width,
    canvas.height
  )
  return canvas
}

function preprocessImage(source, binarize = false) {
  const canvas = document.createElement('canvas')
  canvas.width = source.width
  canvas.height = source.height
  const ctx = canvas.getContext('2d')
  ctx.drawImage(source, 0, 0)
  const image = ctx.getImageData(0, 0, canvas.width, canvas.height)
  const data = image.data
  const histogram = new Uint32Array(256)
  for (let i = 0; i < data.length; i += 4) {
    const gray = Math.max(0, Math.min(255, Math.round(
      (data[i] * 0.299 + data[i + 1] * 0.587 + data[i + 2] * 0.114 - 128) * 1.2 + 128
    )))
    data[i] = data[i + 1] = data[i + 2] = gray
    histogram[gray] += 1
  }

  if (binarize) {
    const total = canvas.width * canvas.height
    let sum = 0
    for (let i = 0; i < 256; i += 1) sum += i * histogram[i]
    let backgroundWeight = 0
    let backgroundSum = 0
    let maxVariance = 0
    let threshold = 128
    for (let i = 0; i < 256; i += 1) {
      backgroundWeight += histogram[i]
      if (!backgroundWeight) continue
      const foregroundWeight = total - backgroundWeight
      if (!foregroundWeight) break
      backgroundSum += i * histogram[i]
      const backgroundMean = backgroundSum / backgroundWeight
      const foregroundMean = (sum - backgroundSum) / foregroundWeight
      const variance = backgroundWeight * foregroundWeight * (backgroundMean - foregroundMean) ** 2
      if (variance > maxVariance) {
        maxVariance = variance
        threshold = i
      }
    }
    for (let i = 0; i < data.length; i += 4) {
      const value = data[i] > threshold ? 255 : 0
      data[i] = data[i + 1] = data[i + 2] = value
    }
  }
  ctx.putImageData(image, 0, 0)
  return canvas
}

async function recognizeInBrowser(frame) {
  const { createWorker } = await import('tesseract.js')
  const worker = await createWorker('eng')
  try {
    let best = null
    for (const canvas of [preprocessImage(frame), preprocessImage(frame, true)]) {
      for (const pageSegMode of ['6', '8']) {
        await worker.setParameters({
          tessedit_pageseg_mode: pageSegMode,
          preserve_interword_spaces: '1'
        })
        const { data } = await worker.recognize(canvas)
        if (!best || (data.confidence ?? 0) > (best.confidence ?? 0)) best = data
      }
    }
    return {
      text: best?.text?.trim() ?? '',
      confidence: best?.confidence ?? 0,
      primary_engine: 'Tesseract.js local fallback',
      needs_review: (best?.confidence ?? 0) < 55,
      engines: { paddle: null, easy: null }
    }
  } finally {
    await worker.terminate()
  }
}

function dataUrlToBlob(dataUrl) {
  const [header, encoded] = dataUrl.split(',')
  const mimeType = header.match(/data:(.*?);base64/)?.[1] ?? 'image/jpeg'
  const binary = atob(encoded)
  const bytes = Uint8Array.from(binary, (char) => char.charCodeAt(0))
  return new Blob([bytes], { type: mimeType })
}

function App() {
  const videoRef = useRef(null)
  const streamRef = useRef(null)
  const [cameraActive, setCameraActive] = useState(false)
  const [facingMode, setFacingMode] = useState('environment')
  const [smallTextMode, setSmallTextMode] = useState(false)
  const [status, setStatus] = useState('Camera is not started.')
  const [processing, setProcessing] = useState(false)
  const [recognizedText, setRecognizedText] = useState('')
  const [capturedImage, setCapturedImage] = useState('')
  const [captureBlob, setCaptureBlob] = useState(null)
  const [recognitionResult, setRecognitionResult] = useState(null)
  const [backendAvailable, setBackendAvailable] = useState(false)
  const [paddleAvailable, setPaddleAvailable] = useState(false)
  const [paddleModelsReady, setPaddleModelsReady] = useState(false)
  const [paddleError, setPaddleError] = useState('')
  const [geminiConsent, setGeminiConsent] = useState(false)
  const [geminiLoading, setGeminiLoading] = useState(false)
  const [geminiAnalysis, setGeminiAnalysis] = useState('')
  const [speed, setSpeed] = useState(1)
  const [error, setError] = useState('')

  const announce = (message) => {
    setStatus(message)
    if ('speechSynthesis' in window) {
      window.speechSynthesis.cancel()
      window.speechSynthesis.speak(new SpeechSynthesisUtterance(message))
    }
  }

  const stopCamera = () => {
    streamRef.current?.getTracks().forEach((track) => track.stop())
    streamRef.current = null
    if (videoRef.current) videoRef.current.srcObject = null
    setCameraActive(false)
  }

  const startCamera = async (requestedFacingMode = facingMode) => {
    stopCamera()
    setError('')
    announce('Camera is starting.')
    if (!navigator.mediaDevices?.getUserMedia) {
      const message = !window.isSecureContext
        ? 'Camera access on a phone requires HTTPS. Open VisionVoice over a secure HTTPS connection.'
        : 'Camera access is not supported by this browser.'
      setError(message)
      announce(message)
      return
    }
    try {
      const stream = await navigator.mediaDevices.getUserMedia({
        video: { facingMode: { ideal: requestedFacingMode }, width: { ideal: 1280 } },
        audio: false
      })
      streamRef.current = stream
      videoRef.current.srcObject = stream
      await videoRef.current.play()
      setCameraActive(true)
      announce('Camera is ready. Position the text and press Capture and Read.')
    } catch (cameraError) {
      const message = cameraError.name === 'NotAllowedError'
        ? 'Camera permission denied. Allow camera access in your browser settings.'
        : 'Camera could not be started. Check that it is connected and not being used by another application.'
      setError(message)
      announce(message)
    }
  }

  const switchCamera = () => {
    const nextMode = facingMode === 'environment' ? 'user' : 'environment'
    setFacingMode(nextMode)
    startCamera(nextMode)
  }

  const readText = (text = recognizedText) => {
    if (!text) {
      announce('There is no recognized text to read.')
      return
    }
    if (!('speechSynthesis' in window)) {
      setError('Text-to-speech is unavailable in this browser.')
      return
    }
    window.speechSynthesis.cancel()
    const utterance = new SpeechSynthesisUtterance(text)
    utterance.rate = speed
    utterance.lang = 'en-US'
    utterance.onstart = () => setStatus('Reading text.')
    window.speechSynthesis.speak(utterance)
  }

  const recognizeCapture = async (blob = captureBlob) => {
    if (!blob) {
      announce('Capture an image before retrying recognition.')
      return
    }
    setProcessing(true)
    setError('')
    setGeminiAnalysis('')
    announce('Starting local text recognition. Large images may take up to a minute.')
    try {
      let result
      try {
        const formData = new FormData()
        formData.append('image', blob, 'capture.jpg')
        const controller = new AbortController()
        const timeoutId = window.setTimeout(() => controller.abort(), OCR_REQUEST_TIMEOUT_MS)
        let response
        try {
          response = await fetch('/api/ocr/combined', {
            method: 'POST',
            body: formData,
            signal: controller.signal
          })
        } finally {
          window.clearTimeout(timeoutId)
        }
        if (!response.ok) {
          const payload = await response.json().catch(() => ({}))
          const detail = payload.detail
          const message = typeof detail === 'string' ? detail : detail?.message
          throw new Error(message ?? `OCR service returned ${response.status}.`)
        }
        result = await response.json()
        setBackendAvailable(true)
        const paddleWorked = Boolean(result.engines?.paddle)
        setPaddleAvailable(paddleWorked)
        setPaddleModelsReady(paddleWorked)
        setPaddleError(result.engine_errors?.paddle ?? '')
      } catch (apiError) {
        setBackendAvailable(false)
        setPaddleAvailable(false)
        setPaddleModelsReady(false)
        setPaddleError(apiError.message)
        setStatus(apiError.name === 'AbortError'
          ? 'Local OCR request timed out. Using browser OCR fallback.'
          : 'Local AI backend unavailable. Using browser OCR fallback.')
        try {
          const frame = await createImageBitmap(blob)
          const canvas = document.createElement('canvas')
          canvas.width = frame.width
          canvas.height = frame.height
          canvas.getContext('2d').drawImage(frame, 0, 0)
          frame.close()
          result = await recognizeInBrowser(canvas)
          setError(`Local PaddleOCR/EasyOCR unavailable. Browser Tesseract fallback used. ${apiError.message}`)
        } catch (fallbackError) {
          console.error(fallbackError)
          setError(`OCR failed. ${apiError.message}`)
        }
      }
      if (result) {
        setRecognitionResult(result)
        setRecognizedText(result.text ?? '')
        if (!result.text?.trim()) {
          setError('No readable text was found. Hold the page steady, move closer, and retry.')
          announce('No readable text was found. Please adjust the page and retry.')
        } else if (result.needs_review) {
          announce(`Some text is uncertain. OCR confidence ${Math.round(result.confidence ?? 0)} percent. Please review the displayed text before reading.`)
        } else {
          setStatus(`Text recognized with ${Math.round(result.confidence ?? 0)} percent confidence.`)
          readText(result.text)
        }
      }
    } finally {
      setProcessing(false)
    }
  }

  const captureAndRecognize = async () => {
    if (!cameraActive || !videoRef.current) {
      announce('Start the camera before capturing an image.')
      return
    }
    setError('')
    setRecognizedText('')
    setRecognitionResult(null)
    setGeminiAnalysis('')
    announce('Capturing image.')
    try {
      const frame = makeCaptureCanvas(videoRef.current, smallTextMode ? 2 : 1)
      const imageUrl = frame.toDataURL('image/jpeg', 0.92)
      const blob = dataUrlToBlob(imageUrl)
      setCapturedImage(imageUrl)
      setCaptureBlob(blob)
      await recognizeCapture(blob)
    } catch (captureError) {
      console.error(captureError)
      setError('Image capture failed. Please check the camera and try again.')
      announce('Image capture failed. Please check the camera and try again.')
    }
  }

  const reviewWithGemini = async () => {
    if (!geminiConsent || !capturedImage || !recognitionResult) {
      announce('Enable Gemini image sharing and capture an image before requesting review.')
      return
    }
    setGeminiLoading(true)
    setGeminiAnalysis('')
    announce('Sending this image to Gemini for optional review.')
    try {
      const response = await fetch('/api/ai/analyze', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          image_base64: capturedImage.split(',')[1],
          mime_type: 'image/jpeg',
          paddle_text: recognitionResult.engines?.paddle?.text ?? '',
          easy_text: recognitionResult.engines?.easy?.text ?? ''
        })
      })
      const payload = await response.json().catch(() => ({}))
      if (!response.ok) throw new Error(payload.detail ?? `Gemini service returned ${response.status}.`)
      setGeminiAnalysis(payload.analysis)
      announce('Gemini review is ready. It is advisory and has not changed the recognized text.')
    } catch (reviewError) {
      setError(reviewError.message)
      announce(`Gemini review failed. Local recognition is unchanged. ${reviewError.message}`)
    } finally {
      setGeminiLoading(false)
    }
  }

  const pauseReading = () => {
    if ('speechSynthesis' in window) {
      window.speechSynthesis.pause()
      announce('Reading paused.')
    }
  }

  const resumeReading = () => {
    if ('speechSynthesis' in window) {
      window.speechSynthesis.resume()
      announce('Reading resumed.')
    }
  }

  const stopReading = () => {
    if ('speechSynthesis' in window) window.speechSynthesis.cancel()
    announce('Reading stopped.')
  }

  useEffect(() => {
    fetch('/api/health')
      .then((response) => response.ok ? response.json() : null)
      .then((health) => {
        setBackendAvailable(Boolean(health?.status === 'ok'))
        setPaddleAvailable(Boolean(health?.paddle_available))
        setPaddleModelsReady(Boolean(health?.paddle_models_ready))
        setPaddleError(health?.paddle_error ?? '')
      })
      .catch((healthError) => {
        setBackendAvailable(false)
        setPaddleAvailable(false)
        setPaddleModelsReady(false)
        setPaddleError(healthError.message)
      })
    const onKeyDown = (event) => {
      if (event.repeat) return
      if (event.key === 'Enter' || event.key === ' ') {
        event.preventDefault()
        document.querySelector('[aria-label="Capture image and read recognized text"]')?.click()
      }
      if (event.key === 'Escape') {
        if ('speechSynthesis' in window) window.speechSynthesis.cancel()
        setStatus('Reading stopped.')
      }
    }
    window.addEventListener('keydown', onKeyDown)
    return () => {
      window.removeEventListener('keydown', onKeyDown)
      stopCamera()
      if ('speechSynthesis' in window) window.speechSynthesis.cancel()
    }
  }, [])

  return (
    <Container maxWidth="md" className="app-shell">
      <header className="brand">
        <CameraAltIcon fontSize="large" aria-hidden="true" />
        <div>
          <Typography component="h1" variant="h4">VisionVoice</Typography>
          <Typography variant="subtitle1">Camera-based reading assistant</Typography>
        </div>
      </header>

      <div className="live-status" role="status" aria-live="assertive">
        {status}
      </div>

      {error && <Alert severity="error" sx={{ mb: 2 }}>{error}</Alert>}
      <Alert
        severity={
          !backendAvailable || (!paddleAvailable && !paddleError && paddleModelsReady)
            ? 'info'
            : paddleAvailable
              ? 'success'
              : 'warning'
        }
        sx={{ mb: 2 }}
      >
        {!backendAvailable
          ? 'Local OCR backend is not connected. Browser Tesseract remains available as a fallback.'
          : paddleAvailable
            ? 'Local PaddleOCR and EasyOCR backend is connected. Images stay on this device unless you explicitly request Gemini review.'
            : paddleError
              ? `OCR API is connected, but PaddleOCR failed. The result may use EasyOCR only. ${paddleError}`
              : paddleModelsReady
                ? 'OCR API is connected. Paddle models are installed, but the runtime has not been verified yet; the first scan will test it. EasyOCR is also available.'
                : 'OCR API is connected, but Paddle model files are unavailable. The result may use EasyOCR only.'}
      </Alert>

      <section aria-label="Camera preview" className={`camera-panel${smallTextMode ? ' small-text-mode' : ''}`}>
        <video ref={videoRef} muted playsInline aria-label="Live camera preview" />
        {!cameraActive && (
          <div className="camera-placeholder">
            <CameraAltIcon sx={{ fontSize: 64 }} aria-hidden="true" />
            <Typography>Camera is off</Typography>
          </div>
        )}
      </section>
      <FormControlLabel
        control={
          <Checkbox
            checked={smallTextMode}
            onChange={(event) => {
              const enabled = event.target.checked
              setSmallTextMode(enabled)
              announce(enabled
                ? 'Small-text mode is on. Aim the text at the center of the camera.'
                : 'Small-text mode is off. The full camera view will be captured.')
            }}
            disabled={processing}
          />
        }
        label="Small-text mode (2× center zoom)"
      />
      {smallTextMode && (
        <Typography variant="body2" sx={{ mb: 1 }}>
          Aim the small print at the center of the camera. This mode captures the center half; move across the page and capture each section.
        </Typography>
      )}

      {capturedImage && (
        <section className="captured-panel" aria-label="Last captured image">
          <Typography component="h2" variant="h6">Last Captured Image</Typography>
          <img src={capturedImage} alt="The text frame used for OCR" />
        </section>
      )}

      <Stack direction={{ xs: 'column', sm: 'row' }} spacing={2} sx={{ my: 2 }}>
        <Button
          variant="contained" size="large" startIcon={<CameraAltIcon />}
          onClick={() => startCamera()} disabled={processing}
          aria-label="Start camera"
        >
          Start Camera
        </Button>
        <Button
          variant="contained" color="secondary" size="large" startIcon={<CameraAltIcon />}
          onClick={captureAndRecognize} disabled={!cameraActive || processing}
          aria-label="Capture image and read recognized text"
        >
          Capture & Read
        </Button>
        <Button
          variant="outlined" size="large" startIcon={<SwitchCameraIcon />}
          onClick={switchCamera} disabled={!cameraActive || processing}
        >
          Switch Camera
        </Button>
      </Stack>

      {processing && (
        <Box sx={{ mb: 2 }} role="status" aria-live="polite">
          <LinearProgress aria-label="OCR processing" />
          <Typography sx={{ mt: 1 }}>Processing locally with PaddleOCR and EasyOCR. Large images may take about a minute on CPU; if the backend does not respond within 90 seconds, browser OCR will be tried.</Typography>
        </Box>
      )}

      <Stack direction="row" spacing={1} alignItems="center" sx={{ mb: 1 }}>
        <Typography component="h2" variant="h5">Recognized Text</Typography>
        {recognitionResult && (
          <Chip label={`${Math.round(recognitionResult.confidence ?? 0)}% confidence`} color={recognitionResult.needs_review ? 'warning' : 'success'} />
        )}
      </Stack>
      <TextField
        multiline
        minRows={7}
        fullWidth
        value={recognizedText}
        onChange={(event) => setRecognizedText(event.target.value)}
        placeholder="Captured text will appear here."
        slotProps={{ htmlInput: { 'aria-label': 'Recognized text' } }}
      />

      <Stack direction={{ xs: 'column', sm: 'row' }} spacing={1.5} sx={{ my: 2 }}>
        <Button variant="outlined" startIcon={<ReplayIcon />} onClick={() => recognizeCapture()} disabled={!captureBlob || processing}>
          Retry Recognition
        </Button>
        <Button variant="contained" startIcon={<ReplayIcon />} onClick={() => readText()}>
          Read Recognized Text
        </Button>
        <Button variant="outlined" startIcon={<PauseIcon />} onClick={pauseReading}>
          Pause
        </Button>
        <Button variant="outlined" startIcon={<PlayArrowIcon />} onClick={resumeReading}>
          Resume
        </Button>
        <Button variant="outlined" color="error" startIcon={<StopIcon />} onClick={stopReading}>
          Stop Reading
        </Button>
      </Stack>

      {recognitionResult && (
        <>
          {recognitionResult.needs_review && (
            <Alert severity="warning" sx={{ mb: 2 }}>
              OCR engines disagree or found low-confidence text. Review the original results before relying on this text.
            </Alert>
          )}
          {recognitionResult.image_diagnostics && (
            <Typography variant="body2" sx={{ mb: 1 }}>
              Image quality: {recognitionResult.image_diagnostics.is_blurry ? 'blurry' : 'clear'}; brightness {recognitionResult.image_diagnostics.brightness}/255.
            </Typography>
          )}
          {recognitionResult.uncertain_regions?.length > 0 && (
            <Box component="section" aria-label="Uncertain OCR regions" sx={{ mb: 2 }}>
              <Typography component="h3" variant="h6">Uncertain sections</Typography>
              {recognitionResult.uncertain_regions.map((region, index) => (
                <Alert severity="warning" key={`${region.bbox?.join(',')}-${index}`} sx={{ mb: 1 }}>
                  PaddleOCR: {region.paddle_text ?? 'No match'}; EasyOCR: {region.easy_text ?? 'No match'}
                </Alert>
              ))}
            </Box>
          )}
          <Accordion>
            <AccordionSummary expandIcon={<ExpandMoreIcon />}>Original OCR engine results</AccordionSummary>
            <AccordionDetails>
              <Typography component="h3" variant="h6">PaddleOCR</Typography>
              <pre className="ocr-result">{recognitionResult.engines?.paddle?.text || 'No PaddleOCR result.'}</pre>
              <Typography component="h3" variant="h6">EasyOCR</Typography>
              <pre className="ocr-result">{recognitionResult.engines?.easy?.text || 'No EasyOCR result.'}</pre>
            </AccordionDetails>
          </Accordion>
        </>
      )}

      <Box sx={{ mt: 2, p: 2, border: '1px solid #cbd5e1', borderRadius: 2, background: 'white' }}>
        <FormControlLabel
          control={
            <Checkbox
              checked={geminiConsent}
              onChange={(event) => setGeminiConsent(event.target.checked)}
            />
          }
          label="I consent to send this captured image and OCR text to Google Gemini for review."
        />
        <Typography variant="body2" sx={{ mb: 1 }}>
          Optional cloud review. The captured image is sent to Google only when you enable consent and press the review button. Gemini suggestions never replace local OCR automatically.
        </Typography>
        <Button
          variant="outlined"
          onClick={reviewWithGemini}
          disabled={!geminiConsent || !recognitionResult || geminiLoading}
        >
          {geminiLoading ? 'Reviewing with Gemini…' : 'Ask Gemini to Review'}
        </Button>
        {geminiAnalysis && (
          <Alert severity="info" sx={{ mt: 1 }}>
            <strong>Gemini advisory only:</strong>
            <pre className="ocr-result">{geminiAnalysis}</pre>
          </Alert>
        )}
      </Box>

      <Box className="settings-panel">
        <FormControl fullWidth>
          <InputLabel id="language-label">OCR language</InputLabel>
          <Select labelId="language-label" value="eng" label="OCR language" disabled>
            <MenuItem value="eng">English</MenuItem>
          </Select>
        </FormControl>
        <Box>
          <Typography id="speed-label">Speech speed: {speed.toFixed(1)}x</Typography>
          <Slider
            value={speed}
            min={0.5}
            max={2}
            step={0.1}
            onChange={(_, value) => setSpeed(value)}
            aria-labelledby="speed-label"
          />
        </Box>
      </Box>

      <Typography variant="body2" sx={{ mt: 2 }}>
        Keyboard: Press Enter or Space to capture. Press Escape to stop reading.
      </Typography>
    </Container>
  )
}

export default App
