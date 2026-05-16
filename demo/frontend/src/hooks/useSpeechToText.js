import { useState, useRef, useCallback, useEffect } from 'react'

/**
 * Speech-to-Text Hook using Hugging Face Transformers (Whisper tiny.en)
 * 
 * Runs entirely in the browser — no server calls for transcription.
 * Model: Xenova/whisper-tiny.en (~40MB, loaded on first use)
 * Uses WebGPU if available, falls back to WASM.
 */

let pipeline = null

export const useSpeechToText = (options = {}) => {
  const {
    model = 'Xenova/whisper-tiny.en',
    onTranscript,
    onError,
    silenceThreshold = 0.01,
    silenceTimeout = 2000,
    onSilenceDetected,
  } = options

  const [isListening, setIsListening] = useState(false)
  const [isLoading, setIsLoading] = useState(false)
  const [isModelLoading, setIsModelLoading] = useState(false)
  const [modelProgress, setModelProgress] = useState(0)
  const [transcript, setTranscript] = useState('')
  const [error, setError] = useState(null)
  const [recordingDuration, setRecordingDuration] = useState(0)

  const transcriberRef = useRef(null)
  const mediaRecorderRef = useRef(null)
  const audioChunksRef = useRef([])
  const streamRef = useRef(null)
  const recordingStartTimeRef = useRef(null)
  const durationIntervalRef = useRef(null)
  const analyserRef = useRef(null)
  const audioContextRef = useRef(null)
  const silenceStartRef = useRef(null)
  const silenceCheckIntervalRef = useRef(null)
  const hasSpokenRef = useRef(false)

  const isSupported = typeof window !== 'undefined' &&
    typeof navigator !== 'undefined' &&
    !!navigator.mediaDevices &&
    !!navigator.mediaDevices.getUserMedia &&
    typeof AudioContext !== 'undefined'

  const initializeModel = useCallback(async () => {
    if (transcriberRef.current) return

    try {
      setIsModelLoading(true)
      setError(null)

      if (!pipeline) {
        const transformers = await import('@huggingface/transformers')
        pipeline = transformers.pipeline
      }

      transcriberRef.current = await pipeline(
        'automatic-speech-recognition',
        model,
        {
          device: 'webgpu',
          progress_callback: (progress) => {
            if (progress.status === 'progress' && progress.progress != null) {
              setModelProgress(Math.round(progress.progress))
            }
          },
        }
      )
    } catch (err) {
      const errorMessage = err instanceof Error ? err.message : 'Failed to load speech recognition model'
      setError(errorMessage)
      if (onError) onError(err instanceof Error ? err : new Error(errorMessage))
    } finally {
      setIsModelLoading(false)
    }
  }, [model, onError])

  const processAudioBlob = useCallback(async (blob) => {
    const arrayBuffer = await blob.arrayBuffer()
    const audioContext = new AudioContext()
    const audioBuffer = await audioContext.decodeAudioData(arrayBuffer)

    let audioData = audioBuffer.getChannelData(0)

    // Resample to 16kHz if needed
    if (audioBuffer.sampleRate !== 16000) {
      const ratio = audioBuffer.sampleRate / 16000
      const newLength = Math.round(audioData.length / ratio)
      const result = new Float32Array(newLength)
      for (let i = 0; i < newLength; i++) {
        result[i] = audioData[Math.round(i * ratio)]
      }
      audioData = result
    }

    await audioContext.close()
    return audioData
  }, [])

  const transcribeChunk = useCallback(async (audioBlob) => {
    if (!transcriberRef.current) return

    try {
      setIsLoading(true)
      const audioData = await processAudioBlob(audioBlob)

      const result = await transcriberRef.current(audioData, {
        return_timestamps: false,
        chunk_length_s: 30,
        stride_length_s: 5,
      })

      const text = result.text.trim()
      if (text) {
        setTranscript(prev => {
          const newTranscript = prev ? `${prev} ${text}` : text
          if (onTranscript) onTranscript(newTranscript)
          return newTranscript
        })
      }
    } catch (err) {
      const errorMessage = err instanceof Error ? err.message : 'Failed to transcribe audio'
      setError(errorMessage)
      if (onError) onError(err instanceof Error ? err : new Error(errorMessage))
    } finally {
      setIsLoading(false)
    }
  }, [processAudioBlob, onTranscript, onError])

  const processAccumulatedChunks = useCallback(async () => {
    if (audioChunksRef.current.length === 0) return

    const audioBlob = new Blob(audioChunksRef.current, { type: 'audio/webm' })
    audioChunksRef.current = []

    await transcribeChunk(audioBlob)
  }, [transcribeChunk])

  const startListening = useCallback(async () => {
    if (!isSupported) {
      setError('Speech recognition is not supported in this browser')
      return
    }

    try {
      setError(null)

      if (!transcriberRef.current) {
        await initializeModel()
      }

      const stream = await navigator.mediaDevices.getUserMedia({
        audio: {
          channelCount: 1,
          sampleRate: 16000,
          echoCancellation: true,
          noiseSuppression: true,
        }
      })

      streamRef.current = stream

      const mimeType = MediaRecorder.isTypeSupported('audio/webm;codecs=opus')
        ? 'audio/webm;codecs=opus'
        : 'audio/webm'

      const mediaRecorder = new MediaRecorder(stream, {
        mimeType,
        audioBitsPerSecond: 128000,
      })

      mediaRecorderRef.current = mediaRecorder
      audioChunksRef.current = []

      mediaRecorder.ondataavailable = (event) => {
        if (event.data.size > 0) {
          audioChunksRef.current.push(event.data)
        }
      }

      mediaRecorder.onstop = async () => {
        await processAccumulatedChunks()
      }

      mediaRecorder.start()
      setIsListening(true)
      hasSpokenRef.current = false

      // Silence detection via Web Audio API
      const audioContext = new AudioContext()
      audioContextRef.current = audioContext
      const source = audioContext.createMediaStreamSource(stream)
      const analyser = audioContext.createAnalyser()
      analyser.fftSize = 2048
      source.connect(analyser)
      analyserRef.current = analyser
      silenceStartRef.current = null

      const dataArray = new Float32Array(analyser.fftSize)
      silenceCheckIntervalRef.current = setInterval(() => {
        if (!analyserRef.current) return
        analyserRef.current.getFloatTimeDomainData(dataArray)
        let sum = 0
        for (let i = 0; i < dataArray.length; i++) {
          sum += dataArray[i] * dataArray[i]
        }
        const rms = Math.sqrt(sum / dataArray.length)

        if (rms > silenceThreshold) {
          hasSpokenRef.current = true
          silenceStartRef.current = null
        } else if (hasSpokenRef.current) {
          if (!silenceStartRef.current) {
            silenceStartRef.current = Date.now()
          } else if (Date.now() - silenceStartRef.current >= silenceTimeout) {
            if (onSilenceDetected) onSilenceDetected()
          }
        }
      }, 100)

      // Duration tracking
      recordingStartTimeRef.current = Date.now()
      setRecordingDuration(0)
      durationIntervalRef.current = setInterval(() => {
        if (recordingStartTimeRef.current) {
          setRecordingDuration(Math.floor((Date.now() - recordingStartTimeRef.current) / 1000))
        }
      }, 100)

    } catch (err) {
      const errorMessage = err instanceof Error ? err.message : 'Failed to start speech recognition'
      setError(errorMessage)
      if (onError) onError(err instanceof Error ? err : new Error(errorMessage))
    }
  }, [isSupported, initializeModel, processAccumulatedChunks, silenceThreshold, silenceTimeout, onSilenceDetected, onError])

  const stopListening = useCallback(() => {
    if (durationIntervalRef.current) {
      clearInterval(durationIntervalRef.current)
      durationIntervalRef.current = null
    }
    recordingStartTimeRef.current = null

    if (silenceCheckIntervalRef.current) {
      clearInterval(silenceCheckIntervalRef.current)
      silenceCheckIntervalRef.current = null
    }
    silenceStartRef.current = null
    analyserRef.current = null
    if (audioContextRef.current) {
      audioContextRef.current.close().catch(() => {})
      audioContextRef.current = null
    }

    if (mediaRecorderRef.current && mediaRecorderRef.current.state !== 'inactive') {
      mediaRecorderRef.current.stop()
    }

    if (streamRef.current) {
      streamRef.current.getTracks().forEach(track => track.stop())
      streamRef.current = null
    }

    setIsListening(false)
  }, [])

  const resetTranscript = useCallback(() => {
    setTranscript('')
    setError(null)
    setRecordingDuration(0)
  }, [])

  useEffect(() => {
    return () => {
      stopListening()
      if (durationIntervalRef.current) clearInterval(durationIntervalRef.current)
      if (silenceCheckIntervalRef.current) clearInterval(silenceCheckIntervalRef.current)
    }
  }, [stopListening])

  return {
    isListening,
    isLoading,
    isModelLoading,
    modelProgress,
    transcript,
    error,
    recordingDuration,
    startListening,
    stopListening,
    resetTranscript,
    isSupported,
  }
}
