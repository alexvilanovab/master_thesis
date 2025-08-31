import { useD2P } from './D2PContext'
import cn from 'classnames'
import { InferenceSession, Tensor } from 'onnxruntime-web'
import { useEffect, useRef, useState, useCallback } from 'react'
import * as Tone from 'tone'

type D2PProps = {
  showDescriptorKnobs?: boolean
  showControls?: boolean
  d0?: number
  d1?: number
  d2?: number
  d3?: number
  d4?: number
  tempo?: number
  metronome?: boolean
  velocity?: boolean
  id: string
  onDescriptorChange?: (values: { d0: number; d1: number; d2: number; d3: number; d4: number }) => void
  onPatternChange?: (pattern: number[]) => void
}

let session: InferenceSession | null = null

export default function D2P(props: D2PProps) {
  const { currentlyPlayingId, setCurrentlyPlayingId, onStopAll, markD2PAsPlayed } = useD2P()

  const [d0, setD0] = useState(props.d0 ?? 0)
  const [d1, setD1] = useState(props.d1 ?? 0)
  const [d2, setD2] = useState(props.d2 ?? 0)
  const [d3, setD3] = useState(props.d3 ?? 0)
  const [d4, setD4] = useState(props.d4 ?? 0)

  const labels = ['onset_count', 'start', 'center', 'syncopation', 'balance']

  const [tempo, setTempo] = useState(props.tempo ?? 120)
  const [metronome, setMetronome] = useState(props.metronome ?? true)
  const [velocity, setVelocity] = useState(props.velocity ?? true)
  const [pattern, setPattern] = useState<number[]>(new Array(16).fill(0))

  const [audioStarted, setAudioStarted] = useState<'false' | 'true' | 'wait'>('false')
  const [playing, setPlaying] = useState(false)
  const [step, setStep] = useState(-1)

  const tom = useRef<Tone.Sampler | null>(null)
  const click = useRef<Tone.Sampler | null>(null)
  const bell = useRef<Tone.Sampler | null>(null)
  const currentPattern = useRef<number[]>(pattern)
  const currentMetronome = useRef(metronome)
  const currentVelocity = useRef(velocity)
  const transportEventId = useRef<number | null>(null)

  useEffect(() => {
    onStopAll(() => {
      setPlaying(false)
      setStep(-1)
    })
  }, [onStopAll])

  const predict = async () => {
    try {
      if (!session) {
        session = await InferenceSession.create('models/model.onnx')
      }

      // // d1: start, d2: center; this constrain keeps start to be > than center
      // if (d2 < d1) setD2(d1)
      // if (d1 > d2) setD1(d2)

      const realD2 = d2 < d1 ? d1 : d2
      const realD1 = d1 > realD2 ? realD2 : d1

      const d2f = (value: number) => value / 127.0
      const input = [d2f(d0), d2f(realD1), d2f(realD2), d2f(d3), d2f(d4)]

      const inputTensor = new Tensor('float32', Float32Array.from(input), [1, 5])
      const results = await session.run({ 'onnx::Gemm_0': inputTensor })
      const outputTensor = results[Object.keys(results)[0]]

      const output = Array.from(outputTensor.data as Float32Array).map((val) => Math.max(0, Math.min(1, val)))
      currentPattern.current = output
      setPattern(output)
    } catch (error) {
      console.error('Error in predict:', error)
    }
  }

  // Initialize ONNX session and run first prediction
  useEffect(() => {
    const init = async () => {
      if (!session) session = await InferenceSession.create('models/model.onnx')
      await predict()
    }
    init()

    return () => {
      if (session) session = null
    }
  }, [props.id])

  // Update internal state when props change
  useEffect(() => {
    setD0(props.d0 ?? 0)
    setD1(props.d1 ?? 0)
    setD2(props.d2 ?? 0)
    setD3(props.d3 ?? 0)
    setD4(props.d4 ?? 0)
    predict()
  }, [props.d0, props.d1, props.d2, props.d3, props.d4])

  // Update internal state when descriptors change from slider interaction
  useEffect(() => {
    predict()
    props.onDescriptorChange?.({ d0, d1, d2, d3, d4 })
  }, [d0, d1, d2, d3, d4])

  // Notify parent of pattern changes
  useEffect(() => {
    props.onPatternChange?.(pattern)
  }, [pattern])

  const handleStep = useCallback((t: number) => {
    setStep((p) => {
      const next = (p + 1) % 16

      if (!tom.current || !click.current || !bell.current) return next

      if (currentVelocity.current && currentPattern.current[p] > 0)
        tom.current.triggerAttack('C4', t, currentPattern.current[p])
      else if (!currentVelocity.current && currentPattern.current[p] > 0.5) tom.current.triggerAttack('C4', t, 1)

      if (currentMetronome.current && p === 0) bell.current.triggerAttack('C4', t)
      else if (currentMetronome.current && p % 4 === 0) click.current.triggerAttack('C4', t)

      return next
    })
  }, [])

  const startLoop = useCallback(() => {
    if (transportEventId.current !== null) Tone.getTransport().clear(transportEventId.current)
    transportEventId.current = Tone.getTransport().scheduleRepeat(handleStep, '16n', 0)
    Tone.getTransport().start()
    setPlaying(true)
    setCurrentlyPlayingId(props.id)
    markD2PAsPlayed(props.id)
  }, [handleStep, props.id, setCurrentlyPlayingId, markD2PAsPlayed])

  const stopLoop = useCallback(() => {
    if (transportEventId.current !== null) {
      Tone.getTransport().clear(transportEventId.current)
      transportEventId.current = null
    }
    setPlaying(false)
    if (currentlyPlayingId === props.id) setCurrentlyPlayingId(null)
  }, [currentlyPlayingId, props.id, setCurrentlyPlayingId])

  useEffect(() => {
    currentPattern.current = pattern
  }, [pattern])

  useEffect(() => {
    currentMetronome.current = metronome
  }, [metronome])

  useEffect(() => {
    currentVelocity.current = velocity
  }, [velocity])

  useEffect(() => {
    tom.current = new Tone.Sampler({ urls: { C4: 'sounds/tom.wav' }, volume: -8 }).toDestination()
    click.current = new Tone.Sampler({ urls: { C4: 'sounds/click.wav' }, volume: -8 }).toDestination()
    bell.current = new Tone.Sampler({ urls: { C4: 'sounds/bell.wav' }, volume: -8 }).toDestination()
    return () => {
      tom.current?.dispose()
      click.current?.dispose()
      bell.current?.dispose()
      if (transportEventId.current !== null) Tone.getTransport().clear(transportEventId.current)
    }
  }, [])

  useEffect(() => {
    Tone.getTransport().bpm.value = tempo
  }, [tempo])

  useEffect(() => {
    if (currentlyPlayingId !== null && currentlyPlayingId !== props.id && playing) stopLoop()
  }, [currentlyPlayingId, props.id, playing, stopLoop])

  async function playStop() {
    if (audioStarted === 'false') {
      setAudioStarted('wait')
      Tone.start()
      await Tone.loaded()
      Tone.getTransport().bpm.value = tempo
      setAudioStarted('true')
      startLoop()
    } else {
      if (playing) stopLoop()
      else startLoop()
    }
  }

  return (
    <div className="flex flex-col items-center gap-4">
      {props.showDescriptorKnobs && (
        <div className="flex gap-2">
          {labels.map((label, index) => (
            <div key={label} className="flex flex-col items-center">
              <label htmlFor={`descriptor-${index}`}>{label}</label>
              <input
                id={`descriptor-${index}`}
                type="range"
                min="0"
                max="127"
                value={[d0, d1, d2, d3, d4][index]}
                onChange={(e) => {
                  const value = parseInt(e.target.value)
                  if (index === 0) setD0(value)
                  if (index === 1) setD1(value)
                  if (index === 2) setD2(value)
                  if (index === 3) setD3(value)
                  if (index === 4) setD4(value)
                }}
              />
            </div>
          ))}
        </div>
      )}

      {props.showControls && (
        <div className="flex items-center gap-4">
          <div className="flex flex-col items-center">
            <label htmlFor="tempo">{`tempo`}</label>
            <input
              type="range"
              min="60"
              max="180"
              value={tempo}
              onChange={(e) => setTempo(parseInt(e.target.value))}
              id="tempo"
            />
          </div>
          <div className="flex items-center gap-2">
            <div className="flex flex-col items-center">
              <label htmlFor="metronome">{`metronome`}</label>
              <input
                type="checkbox"
                checked={metronome}
                onChange={(e) => setMetronome(e.target.checked)}
                id="metronome"
              />
            </div>
            <div className="flex flex-col items-center">
              <label htmlFor="velocity">{`velocity`}</label>
              <input type="checkbox" checked={velocity} onChange={(e) => setVelocity(e.target.checked)} id="velocity" />
            </div>
          </div>
        </div>
      )}

      <div className="flex gap-1">
        <button
          className={cn(
            'h-10 w-10 cursor-pointer rounded border py-0.5 text-xs',
            playing
              ? 'border-green-300 bg-green-50 hover:bg-green-100'
              : 'border-gray-300 bg-gray-50 hover:bg-gray-100',
          )}
          onClick={() => playStop()}
        >
          {playing ? 'stop' : 'play'}
        </button>

        {pattern.map((value, i) => (
          <div
            key={i}
            className={cn(
              'flex h-10 w-10 items-center justify-center rounded border border-gray-300 text-xs font-bold',
              i === step ? 'border-3 border-red-300' : '',
              value >= 0.5 ? 'text-white' : 'text-black',
            )}
            style={
              velocity
                ? {
                    background: `linear-gradient(to bottom, rgba(255,255,255,1) ${(1 - value) * 100}%, rgba(60,60,60,1) ${(1 - value) * 100}%)`,
                  }
                : { backgroundColor: value >= 0.5 ? 'rgba(60,60,60,1)' : 'white' }
            }
          >
            {/* {value.toFixed(2)} */}
          </div>
        ))}
      </div>
    </div>
  )
}
