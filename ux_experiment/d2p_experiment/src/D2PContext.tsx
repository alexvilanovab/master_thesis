import { createContext, useContext, useState, useCallback } from 'react'
import type { ReactNode } from 'react'
import * as Tone from 'tone'

type D2PContextType = {
  currentlyPlayingId: string | null
  setCurrentlyPlayingId: (id: string | null) => void
  stopAll: () => void
  onStopAll: (callback: () => void) => void
  playedD2Ps: Set<string>
  markD2PAsPlayed: (id: string) => void
}

const D2PContext = createContext<D2PContextType | null>(null)

export function D2PProvider({ children }: { children: ReactNode }) {
  const [currentlyPlayingId, setCurrentlyPlayingId] = useState<string | null>(null)
  const [stopCallbacks, setStopCallbacks] = useState<(() => void)[]>([])
  const [playedD2Ps, setPlayedD2Ps] = useState<Set<string>>(new Set())

  const onStopAll = useCallback((callback: () => void) => {
    setStopCallbacks((prev) => [...prev, callback])
  }, [])

  const stopAll = useCallback(() => {
    Tone.getTransport().stop()
    Tone.getTransport().cancel()
    setCurrentlyPlayingId(null)
    stopCallbacks.forEach((callback) => callback())
  }, [stopCallbacks])

  const markD2PAsPlayed = useCallback((id: string) => {
    setPlayedD2Ps((prev) => {
      const next = new Set(prev)
      next.add(id)
      return next
    })
  }, [])

  return (
    <D2PContext.Provider
      value={{ currentlyPlayingId, setCurrentlyPlayingId, stopAll, onStopAll, playedD2Ps, markD2PAsPlayed }}
    >
      {children}
    </D2PContext.Provider>
  )
}

export function useD2P() {
  const context = useContext(D2PContext)
  if (!context) {
    throw new Error('useD2P must be used within a D2PProvider')
  }
  return context
}
