'use client';

import React, { useEffect, useMemo, useRef } from 'react';
import { Track } from 'livekit-client';
import { AnimatePresence, motion } from 'motion/react';
import {
  type TrackReference,
  VideoTrack,
  useLocalParticipant,
  useTracks,
  useVoiceAssistant,
} from '@livekit/components-react';
import { cn } from '@/lib/utils';

const MotionContainer = motion.create('div');

const ANIMATION_TRANSITION = {
  type: 'spring',
  stiffness: 800,
  damping: 50,
  mass: 1,
};

const classNames = {
  grid: [
    'h-full w-full',
    'grid gap-x-4 place-content-center',
    'grid-cols-[1fr_1fr] grid-rows-[60px_1fr_60px]',
  ],
  agentChatOpenWithSecondTile: ['col-start-1 row-start-1', 'self-center justify-self-end'],
  agentChatOpenWithoutSecondTile: ['col-start-1 row-start-1', 'col-span-2', 'place-content-center'],
  agentChatClosed: ['col-start-1 row-start-1', 'col-span-2 row-span-3', 'place-content-center'],
  secondTileChatOpen: ['col-start-2 row-start-1', 'self-center justify-self-start'],
  secondTileChatClosed: ['col-start-2 row-start-3', 'place-content-end'],
};

export function useLocalTrackRef(source: Track.Source) {
  const { localParticipant } = useLocalParticipant();
  const publication = localParticipant.getTrackPublication(source);
  const trackRef = useMemo<TrackReference | undefined>(
    () => (publication ? { source, participant: localParticipant, publication } : undefined),
    [source, publication, localParticipant]
  );
  return trackRef;
}

/**
 * Custom ECG/Oscilloscope Visualizer
 */
const ECGVisualizer = ({
  trackRef,
  className,
}: {
  trackRef?: TrackReference;
  className?: string;
}) => {
  const canvasRef = useRef<HTMLCanvasElement>(null);

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas || !trackRef?.publication?.track) return;

    const track = trackRef.publication.track;
    if (!track.mediaStreamTrack) return;

    const stream = new MediaStream([track.mediaStreamTrack]);
    const AudioContextClass = window.AudioContext || (window as any).webkitAudioContext;
    const audioContext = new AudioContextClass();
    const analyser = audioContext.createAnalyser();
    const source = audioContext.createMediaStreamSource(stream);

    analyser.fftSize = 2048;
    source.connect(analyser);

    const bufferLength = analyser.frequencyBinCount;
    const dataArray = new Uint8Array(bufferLength);

    let animationId: number;

    const draw = () => {
      animationId = requestAnimationFrame(draw);

      analyser.getByteTimeDomainData(dataArray);
      const ctx = canvas.getContext('2d');
      if (!ctx) return;

      const width = canvas.width;
      const height = canvas.height;

      ctx.clearRect(0, 0, width, height);

      /** Draw waveform */
      ctx.lineWidth = 2;
      ctx.strokeStyle = '#ffffff';
      ctx.shadowBlur = 8;
      ctx.shadowColor = '#ffffff';

      ctx.beginPath();
      const sliceWidth = (width * 1.0) / bufferLength;
      let x = 0;

      for (let i = 0; i < bufferLength; i++) {
        const v = dataArray[i] / 128.0;
        const y = (v * height) / 2;

        i === 0 ? ctx.moveTo(x, y) : ctx.lineTo(x, y);
        x += sliceWidth;
      }

      ctx.lineTo(width, height / 2);
      ctx.stroke();
    };

    draw();

    return () => {
      cancelAnimationFrame(animationId);
      source.disconnect();
      analyser.disconnect();
      audioContext.close();
    };
  }, [trackRef]);

  return <canvas ref={canvasRef} className={className} width={300} height={150} />;
};

interface TileLayoutProps {
  chatOpen: boolean;
}

export function TileLayout({ chatOpen }: TileLayoutProps) {
  const {
    state: agentState,
    audioTrack: agentAudioTrack,
    videoTrack: agentVideoTrack,
  } = useVoiceAssistant();
  const [screenShareTrack] = useTracks([Track.Source.ScreenShare]);
  const cameraTrack: TrackReference | undefined = useLocalTrackRef(Track.Source.Camera);

  const isCameraEnabled = cameraTrack && !cameraTrack.publication.isMuted;
  const isScreenShareEnabled = screenShareTrack && !screenShareTrack.publication.isMuted;
  const hasSecondTile = isCameraEnabled || isScreenShareEnabled;

  const animationDelay = chatOpen ? 0 : 0.15;
  const isAvatar = agentVideoTrack !== undefined;
  const videoWidth = agentVideoTrack?.publication.dimensions?.width ?? 0;
  const videoHeight = agentVideoTrack?.publication.dimensions?.height ?? 0;

  return (
    <div className="pointer-events-none fixed inset-x-0 top-8 bottom-32 z-50 md:top-12 md:bottom-40">
      <div className="relative mx-auto h-full max-w-4xl px-4 md:px-0">
        <div className={cn(classNames.grid)}>
          
          {/* =======================
              AGENT TILE (AUDIO/VIDEO)
             ======================= */}
          <div
            className={cn([
              'grid transition-all duration-500 ease-spring',
              !chatOpen && classNames.agentChatClosed,
              chatOpen && hasSecondTile && classNames.agentChatOpenWithSecondTile,
              chatOpen && !hasSecondTile && classNames.agentChatOpenWithoutSecondTile,
            ])}
          >
            <AnimatePresence mode="popLayout">

              {/* AUDIO MODE (ECG TILE) */}
              {!isAvatar && (
                <MotionContainer
                  key="agent"
                  layoutId="agent"
                  initial={{ opacity: 0, scale: 0.8, filter: 'blur(10px)' }}
                  animate={{ opacity: 1, scale: chatOpen ? 1 : 1.2, filter: 'blur(0px)' }}
                  transition={{ ...ANIMATION_TRANSITION, delay: animationDelay }}
                  className={cn(
                    'relative overflow-hidden',
                    'bg-black/95 backdrop-blur-md',
                    'border border-white',
                    'shadow-[0_0_18px_rgba(255,255,255,0.45)]',
                    chatOpen ? 'h-[60px] w-[60px] rounded-lg' : 'h-[120px] w-[120px] rounded-xl'
                  )}
                >
                  <div
                    className="absolute inset-0 z-0 opacity-15"
                    style={{
                      backgroundImage: `linear-gradient(rgba(255,255,255,0.35) 1px, transparent 1px),
                                        linear-gradient(90deg, rgba(255,255,255,0.35) 1px, transparent 1px)`,
                      backgroundSize: '10px 10px',
                    }}
                  />

                  <ECGVisualizer
                    trackRef={agentAudioTrack}
                    className="relative z-10 h-full w-full"
                  />
                </MotionContainer>
              )}

              {/* VIDEO MODE (AVATAR TILE) */}
              {isAvatar && (
                <MotionContainer
                  key="avatar"
                  layoutId="avatar"
                  initial={{
                    scale: 1,
                    opacity: 1,
                    maskImage: 'radial-gradient(circle, black 0%, transparent 0%)',
                  }}
                  animate={{
                    maskImage: chatOpen
                      ? 'radial-gradient(circle, black 100%, transparent 100%)'
                      : 'radial-gradient(circle, black 60%, transparent 70%)',
                    borderRadius: chatOpen ? 8 : 12,
                  }}
                  transition={{
                    ...ANIMATION_TRANSITION,
                    delay: animationDelay,
                    maskImage: { duration: 0.8 },
                  }}
                  className={cn(
                    'relative overflow-hidden bg-black',
                    'border border-white shadow-[0_0_18px_rgba(255,255,255,0.40)]',
                    chatOpen
                      ? 'h-[60px] w-[60px]'
                      : 'h-auto w-full max-w-[400px] aspect-video'
                  )}
                >
                  <VideoTrack
                    width={videoWidth}
                    height={videoHeight}
                    trackRef={agentVideoTrack}
                    className={cn(
                      'h-full w-full object-cover opacity-90 grayscale-[0.2]',
                      chatOpen ? 'scale-110' : 'scale-100'
                    )}
                  />
                </MotionContainer>
              )}
            </AnimatePresence>
          </div>

          {/* =======================
              CAMERA / SCREEN TILE
             ======================= */}
          <div
            className={cn([
              'grid transition-all duration-500',
              chatOpen && classNames.secondTileChatOpen,
              !chatOpen && classNames.secondTileChatClosed,
            ])}
          >
            <AnimatePresence>
              {(cameraTrack && isCameraEnabled ||
                screenShareTrack && isScreenShareEnabled) && (
                <MotionContainer
                  key="camera"
                  layout="position"
                  layoutId="camera"
                  initial={{ opacity: 0, scale: 0.8, y: 20 }}
                  animate={{ opacity: 1, scale: 1, y: 0 }}
                  exit={{ opacity: 0, scale: 0.8, y: 20 }}
                  transition={{ ...ANIMATION_TRANSITION, delay: animationDelay }}
                  className={cn(
                    'relative overflow-hidden',
                    'border border-white shadow-[0_0_15px_rgba(255,255,255,0.4)]',
                    'bg-black',
                    'h-[60px] w-[60px] rounded-lg'
                  )}
                >
                  <VideoTrack
                    trackRef={cameraTrack || screenShareTrack}
                    width={(cameraTrack || screenShareTrack)?.publication.dimensions?.width ?? 0}
                    height={(cameraTrack || screenShareTrack)?.publication.dimensions?.height ?? 0}
                    className="h-full w-full object-cover grayscale-[0.1]"
                  />

                  <div className="absolute bottom-1 right-1 h-1.5 w-1.5 rounded-full bg-white shadow-[0_0_6px_white]" />
                </MotionContainer>
              )}
            </AnimatePresence>
          </div>

        </div>
      </div>
    </div>
  );
}
