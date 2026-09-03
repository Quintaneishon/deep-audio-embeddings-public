import React, { useEffect, useRef, useState, useMemo } from 'react';
import * as THREE from 'three';
import { OrbitControls } from 'three/examples/jsm/controls/OrbitControls';
import { EffectComposer } from 'three/examples/jsm/postprocessing/EffectComposer.js';
import { RenderPass } from 'three/examples/jsm/postprocessing/RenderPass.js';
import { UnrealBloomPass } from 'three/examples/jsm/postprocessing/UnrealBloomPass.js';

// ── Tunable constants ──────────────────────────────────────────────────────
const ZOOM_INITIAL      = 1.0;   // 1 — multiplier of TARGET_SCALE for starting camera distance
const ZOOM_RETURN       = 0.5;   // 2 — multiplier of formScale for camera distance when returning to last song
const ZOOM_WAVE         = 2.2;   // 3 — multiplier of formScale for camera distance during wave animation
const POINT_BASE_SIZE   = 4.0;   // 4 — base glass size (pixels at reference distance)
const GLASS_RIM_SHARPNESS = 5.0; // 6 — glass border sharpness (higher = thinner rim)
const BLOOM_THRESHOLD     = 10.0; // 7 — luminance threshold for bloom (only blink exceeds this)
const GLASS_BRIGHTNESS    = 5.8;  // 8 — base brightness multiplier for glass particles
const GLASS_PLAYING_BRIGHTNESS = 1.0; // 9 — brightness multiplier when a song is playing
const POINT_BLINK_MULT  = 0.5;        // 10 — blink size multiplier (pulse peak = POINT_BASE_SIZE * (1 + this))

// ── Colour helpers ─────────────────────────────────────────────────────────
const STATIC_COLORS = [
  [255, 0, 0], [255, 105, 180], [138, 43, 226], [255, 215, 0],
  [0, 255, 255], [255, 140, 0], [34, 139, 34], [0, 0, 255],
  [128, 128, 128], [139, 69, 19],
];
const DEFAULT_COLOR = [200, 200, 200];

const createGenreColorMap = async () => {
  try {
    const res = await fetch(`${process.env.REACT_APP_API_BASE_URL || 'http://127.0.0.1:5000'}/tags`);
    const tags = await res.json();
    const map = {};
    tags.forEach((tag, i) => {
      map[tag.toLowerCase()] = i < STATIC_COLORS.length
        ? STATIC_COLORS[i]
        : [Math.random()*255|0, Math.random()*255|0, Math.random()*255|0];
    });
    return map;
  } catch { return {}; }
};

const getColorForGenre = (genre, map) => {
  if (!genre || !map) return DEFAULT_COLOR;
  return map[genre.toLowerCase()] || DEFAULT_COLOR;
};

const fmt = (t) => {
  if (!t || isNaN(t)) return '0:00';
  return `${Math.floor(t / 60)}:${String(Math.floor(t % 60)).padStart(2, '0')}`;
};

// ── Main component ─────────────────────────────────────────────────────────
export const ParticlesVisualization = () => {
  const mountRef = useRef(null);

  // Three.js refs
  const rendererRef  = useRef(null);
  const composerRef  = useRef(null);
  const cameraRef    = useRef(null);
  const controlsRef  = useRef(null);
  const pointsRef    = useRef(null);  // THREE.Points (glass)
  const clockRef     = useRef(new THREE.Clock());
  const frameRef     = useRef(null);
  const raycasterRef = useRef(new THREE.Raycaster());

  // Particle data refs
  const basePositionsRef = useRef([]);
  const curPositionsRef  = useRef([]);
  const pointsDataRef    = useRef([]);
  const formScaleRef     = useRef(80);

  // Camera fly-to animation ref
  // When play starts, camera lerps from wherever it is → formation view
  const camAnimRef = useRef({
    active: false, t: 0,
    fromPos:    new THREE.Vector3(),
    fromTarget: new THREE.Vector3(),
    toPos:      new THREE.Vector3(),
    toTarget:   new THREE.Vector3(0, 0, 0),
  });

  // Audio refs
  const analyserRef  = useRef(null);
  const audioCtxRef  = useRef(null);
  const audioElRef   = useRef(null);
  const isPlayingRef = useRef(false);
  const ampRef       = useRef(0);
  const fftRef       = useRef(new Uint8Array(256));

  // React state (UI only)
  const [loading,       setLoading]       = useState(true);
  const [error,         setError]         = useState(null);
  const [genreColorMap, setGenreColorMap] = useState(null);
  const [selectedSong,  setSelectedSong]  = useState(null);
  const [isPlaying,     setIsPlaying]     = useState(false);
  const [currentTime,   setCurrentTime]   = useState(0);
  const [duration,      setDuration]      = useState(0);
  const [searchQuery,   setSearchQuery]   = useState('');
  const [embData,       setEmbData]       = useState([]);

  // ── Build Three.js scene ─────────────────────────────────────────────────
  const buildScene = (embeddings, colorMap) => {
    const mount = mountRef.current;
    if (!mount) return () => {};
    const W = mount.clientWidth  || window.innerWidth;
    const H = mount.clientHeight || window.innerHeight;
    const n = embeddings.length;

    // Scene
    const scene = new THREE.Scene();
    scene.fog = new THREE.FogExp2(0x000000, 0.006);

    // Camera
    const camera = new THREE.PerspectiveCamera(60, W / H, 0.1, 2000);
    cameraRef.current = camera;

    // Renderer
    const renderer = new THREE.WebGLRenderer({ antialias: true, powerPreference: 'high-performance' });
    renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
    renderer.setSize(W, H);
    mount.appendChild(renderer.domElement);
    rendererRef.current = renderer;

    // Bloom post-processing
    const composer = new EffectComposer(renderer);
    composer.addPass(new RenderPass(scene, camera));
    const bloom = new UnrealBloomPass(new THREE.Vector2(W, H), 1.5, 0.4, 0.85);
    bloom.strength = 2.2; bloom.radius = 0.6; bloom.threshold = BLOOM_THRESHOLD;
    composer.addPass(bloom);
    composerRef.current = composer;

    // OrbitControls
    const controls = new OrbitControls(camera, renderer.domElement);
    controls.enableDamping = true;
    controls.dampingFactor = 0.05;
    controlsRef.current = controls;

    // ── Normalise UMAP positions into world space ────────────────────────────
    let minX = Infinity, maxX = -Infinity;
    let minY = Infinity, maxY = -Infinity;
    let minZ = Infinity, maxZ = -Infinity;
    embeddings.forEach(e => {
      const [x, y, z = 0] = e.coords;
      if (x < minX) minX = x; if (x > maxX) maxX = x;
      if (y < minY) minY = y; if (y > maxY) maxY = y;
      if (z < minZ) minZ = z; if (z > maxZ) maxZ = z;
    });
    const rangeMax    = Math.max(maxX - minX, maxY - minY, maxZ - minZ) || 1;
    const TARGET_SCALE = 80;
    const norm = (v, min) => ((v - min) / rangeMax) * TARGET_SCALE - TARGET_SCALE / 2;
    formScaleRef.current = TARGET_SCALE * 0.85;

    const base   = [];
    const cur    = [];
    const ptData = [];
    embeddings.forEach((emb) => {
      const [rx, ry, rz = 0] = emb.coords;
      const v = new THREE.Vector3(norm(rx, minX), norm(ry, minY), norm(rz, minZ));
      base.push(v.clone());
      cur.push(v.clone());
      ptData.push({
        name: emb.name, tag: emb.tag, audio: emb.audio,
        genreColor: getColorForGenre(emb.tag, colorMap),
      });
    });
    basePositionsRef.current = base;
    curPositionsRef.current  = cur;
    pointsDataRef.current    = ptData;

    // Camera starts above the UMAP cloud
    camera.position.set(0, 0, TARGET_SCALE * ZOOM_INITIAL);
    controls.target.set(0, 0, 0);
    controls.update();

    // Pre-compute the formation view position
    camAnimRef.current.toPos.set(0, 0, formScaleRef.current * ZOOM_WAVE);
    camAnimRef.current.toTarget.set(0, 0, 0);

    // ── Glass particle buffers ───────────────────────────────────────────────
    const posArr  = new Float32Array(n * 3);
    const colArr  = new Float32Array(n * 3);
    const sizeArr = new Float32Array(n);

    for (let i = 0; i < n; i++) {
      posArr[i * 3]     = cur[i].x;
      posArr[i * 3 + 1] = cur[i].y;
      posArr[i * 3 + 2] = cur[i].z;
      const c = ptData[i].genreColor;
      colArr[i * 3]     = c[0] / 255;
      colArr[i * 3 + 1] = c[1] / 255;
      colArr[i * 3 + 2] = c[2] / 255;
      sizeArr[i] = POINT_BASE_SIZE;
    }

    const geo = new THREE.BufferGeometry();
    geo.setAttribute('position', new THREE.BufferAttribute(posArr,  3));
    geo.setAttribute('color',    new THREE.BufferAttribute(colArr,  3));
    geo.setAttribute('aSize',    new THREE.BufferAttribute(sizeArr, 1));

    const mat = new THREE.ShaderMaterial({
      vertexColors: true,
      transparent:  true,
      depthWrite:   false,
      uniforms: {
        uRimSharpness: { value: GLASS_RIM_SHARPNESS },
        uBrightness:   { value: GLASS_BRIGHTNESS },
      },
      vertexShader: `
        attribute float aSize;
        varying vec3 vColor;
        void main() {
          vColor = color;
          vec4 mv = modelViewMatrix * vec4(position, 1.0);
          gl_PointSize = aSize * (400.0 / -mv.z);
          gl_Position  = projectionMatrix * mv;
        }
      `,
      fragmentShader: `
        uniform float uRimSharpness;
        uniform float uBrightness;
        varying vec3 vColor;
        void main() {
          vec2  uv = 2.0 * gl_PointCoord - 1.0;
          float r  = length(uv);
          if (r > 1.0) discard;
          // Fresnel-like rim: bright at edge, dark at centre (no bloom — stays below threshold)
          float fresnel = clamp(1.0 - (1.0 - r), 0.0, 1.0);
          fresnel = pow(fresnel, uRimSharpness);
          // Faint body fill so the sphere reads as solid glass
          float body = (1.0 - r * r) * 0.06;
          // Small specular dot (top-left) — tinted, not white, stays sub-threshold
          vec2  sv   = uv - vec2(-0.28, 0.32);
          float spec = exp(-dot(sv, sv) * 12.0) * 0.25;
          float alpha = clamp(fresnel * 0.9 + body + spec, 0.0, 1.0);
          // uBrightness scales colour without pushing to white (stays sub-threshold)
          vec3  col  = vColor * uBrightness * (fresnel + body) + vColor * spec;
          gl_FragColor = vec4(col, alpha);
        }
      `,
    });

    const points = new THREE.Points(geo, mat);
    scene.add(points);
    pointsRef.current = points;

    // Raycaster threshold proportional to cloud size
    raycasterRef.current.params.Points = { threshold: TARGET_SCALE * 0.012 };

    // ── Animation loop ───────────────────────────────────────────────────────
    const target = new THREE.Vector3();
    const pColor = new THREE.Color();
    const tau    = Math.PI * 2; // used in wave formation

    const animate = () => {
      frameRef.current = requestAnimationFrame(animate);
      const time = clockRef.current.getElapsedTime();

      // ── Audio energy ──────────────────────────────────────────────────────
      if (analyserRef.current && isPlayingRef.current) {
        analyserRef.current.getByteFrequencyData(fftRef.current);
        const avg = fftRef.current.reduce((a, b) => a + b, 0) / fftRef.current.length / 255;
        ampRef.current = ampRef.current * 0.8 + avg * 0.2;
      } else {
        ampRef.current *= 0.93;
      }
      const amp     = ampRef.current;
      const playing = isPlayingRef.current && amp > 0.005;
      const fft     = fftRef.current;

      // Smoothly ramp brightness up when playing, down when stopped
      const targetBrightness = playing ? GLASS_PLAYING_BRIGHTNESS : GLASS_BRIGHTNESS;
      mat.uniforms.uBrightness.value += (targetBrightness - mat.uniforms.uBrightness.value) * 0.05;

      // FFT bands
      const bass = (fft[0]+fft[1]+fft[2]+fft[3]+fft[4]) / 5 / 255;
      const mids = (() => { let s = 0; for (let k = 10; k < 60; k++) s += fft[k]; return s / 50 / 255; })();
      const high = (() => { let s = 0; for (let k = 80; k < 128; k++) s += fft[k]; return s / 48 / 255; })();

      // Formation params
      const scale     = formScaleRef.current * (0.9 + amp * 0.8);
      const lerpSpeed = playing ? 0.04 + amp * 0.06 : 0.025;

      for (let i = 0; i < n; i++) {
        if (playing) {
          // ── Single wave formation ────────────────────────────────────────
          // Particles spread along X, displaced in Y (and a little Z) by
          // a sine wave whose amplitude and speed are driven by the audio.
          const u = i / n;                                 // 0 → 1 along the line

          const width = scale * 2.2;                       // total X spread
          const x = (u - 0.5) * width;

          // Primary wave: amplitude driven by overall energy, freq by bass
          const waveFreq  = 4.0 + bass * 8.0;             // 4–12 cycles across the line
          const waveSpeed = 2.0 + mids * 4.0;             // travel speed
          const waveAmp   = scale * 0.35 * amp;
          const y = Math.sin(u * waveFreq * tau + time * waveSpeed) * waveAmp;

          // Secondary wave in Z for depth (half freq, driven by highs)
          const z = Math.sin(u * waveFreq * 0.5 * tau - time * waveSpeed * 0.6) * scale * 0.12 * (amp + high * 0.5);

          target.set(x, y, z);

          // Colour: rainbow along the line, lightness pulses with audio
          const hue  = (u + time * 0.05) % 1.0;
          const sat  = 0.85;
          const glow = Math.min(1.0, 0.45 + amp * 0.55 + Math.abs(Math.sin(u * waveFreq * tau + time * waveSpeed)) * amp * 0.3);
          pColor.setHSL(hue, sat, glow);

        } else {
          // Return to UMAP position
          target.copy(basePositionsRef.current[i]);
          const c = ptData[i].genreColor;
          pColor.setRGB(c[0] / 255, c[1] / 255, c[2] / 255);
        }

        curPositionsRef.current[i].lerp(target, lerpSpeed);
        const p = curPositionsRef.current[i];
        posArr[i * 3]     = p.x;
        posArr[i * 3 + 1] = p.y;
        posArr[i * 3 + 2] = p.z;
        colArr[i * 3]     = pColor.r;
        colArr[i * 3 + 1] = pColor.g;
        colArr[i * 3 + 2] = pColor.b;
        // All particles pulse uniformly with global amplitude — no per-bin size variation
        sizeArr[i] = POINT_BASE_SIZE * (1.0 + amp * 2.0);
      }

      // ── Blink last-played point when not playing ─────────────────────────
      const li = lastPlayedIdxRef.current;
      if (!playing && li !== null) {
        const pulse = 0.5 + 0.5 * Math.sin(time * 5.0);       // 2.5 Hz blink
        const lc    = ptData[li].genreColor;
        // Blend genre colour → pure white; white (1,1,1) exceeds BLOOM_THRESHOLD → blooms
        colArr[li * 3]     = lc[0] / 255 + pulse * (1.0 - lc[0] / 255);
        colArr[li * 3 + 1] = lc[1] / 255 + pulse * (1.0 - lc[1] / 255);
        colArr[li * 3 + 2] = lc[2] / 255 + pulse * (1.0 - lc[2] / 255);
        sizeArr[li] = POINT_BASE_SIZE * (1.0 + pulse * POINT_BLINK_MULT);
      }

      geo.attributes.position.needsUpdate = true;
      geo.attributes.color.needsUpdate    = true;
      geo.attributes.aSize.needsUpdate    = true;

      // ── Camera fly-to animation ───────────────────────────────────────────
      const ca = camAnimRef.current;
      if (ca.active) {
        ca.t = Math.min(1.0, ca.t + 0.012); // ~80 frames ≈ 1.3 s
        const eased = 1 - Math.pow(1 - ca.t, 3); // ease-out cubic
        camera.position.lerpVectors(ca.fromPos,    ca.toPos,    eased);
        controls.target.lerpVectors(ca.fromTarget, ca.toTarget, eased);
        controls.update();
        if (ca.t >= 1.0) ca.active = false;
      } else {
        controls.update();
      }

      composer.render();
    };
    animate();

    // Resize
    const onResize = () => {
      const w = mount.clientWidth;
      const h = mount.clientHeight;
      camera.aspect = w / h;
      camera.updateProjectionMatrix();
      renderer.setSize(w, h);
      composer.setSize(w, h);
    };
    window.addEventListener('resize', onResize);
    return () => window.removeEventListener('resize', onResize);
  };

  // ── Load data ────────────────────────────────────────────────────────────
  useEffect(() => {
    let cleanupFn = null;
    const load = async () => {
      try {
        const colorMap = await createGenreColorMap();
        setGenreColorMap(colorMap);
        const res = await fetch(
          `${process.env.REACT_APP_API_BASE_URL || 'http://127.0.0.1:5000'}/embeddings?red=whisper_contrastive&dataset=base&metodo=umap&dimensions=3`
        );
        if (!res.ok) throw new Error(`HTTP ${res.status}`);
        const result = await res.json();
        const embeddings = result.data || [];
        setEmbData(embeddings);
        cleanupFn = buildScene(embeddings, colorMap);
      } catch (err) {
        setError(err.message);
      } finally {
        setLoading(false);
      }
    };
    load();
    return () => {
      if (frameRef.current) cancelAnimationFrame(frameRef.current);
      if (rendererRef.current) {
        const canvas = rendererRef.current.domElement;
        if (canvas.parentNode) canvas.parentNode.removeChild(canvas);
        rendererRef.current.dispose();
      }
      stopAudio();
      if (cleanupFn) cleanupFn();
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  // ESC
  useEffect(() => {
    const fn = (e) => {
      if (e.key === 'Escape') { stopAudio(); flyToSongPos(selectedIdxRef.current); setSelectedSong(null); }
    };
    window.addEventListener('keydown', fn);
    return () => window.removeEventListener('keydown', fn);
  }, []);

  // ── Audio ────────────────────────────────────────────────────────────────
  const stopAudio = () => {
    if (audioElRef.current) { audioElRef.current.pause(); audioElRef.current.src = ''; }
    isPlayingRef.current = false;
    setIsPlaying(false);
  };

  const playAudioUrl = async (url) => {
    stopAudio();
    if (audioCtxRef.current && audioCtxRef.current.state !== 'closed')
      await audioCtxRef.current.close().catch(() => {});

    const ctx      = new (window.AudioContext || window.webkitAudioContext)();
    const analyser = ctx.createAnalyser();
    analyser.fftSize = 512;
    audioCtxRef.current = ctx;
    analyserRef.current = analyser;
    fftRef.current = new Uint8Array(analyser.frequencyBinCount);

    const audio = new Audio(url);
    audio.crossOrigin = 'anonymous';
    audioElRef.current = audio;
    ctx.createMediaElementSource(audio).connect(analyser);
    analyser.connect(ctx.destination);

    audio.addEventListener('loadedmetadata', () => setDuration(audio.duration));
    audio.addEventListener('timeupdate',     () => setCurrentTime(audio.currentTime));
    audio.addEventListener('ended', () => { isPlayingRef.current = false; setIsPlaying(false); });

    await audio.play();
    isPlayingRef.current = true;
    setIsPlaying(true);

    // Trigger camera fly-to formation view
    const ca      = camAnimRef.current;
    const camera  = cameraRef.current;
    const controls = controlsRef.current;
    if (camera && controls) {
      ca.fromPos.copy(camera.position);
      ca.fromTarget.copy(controls.target);
      ca.toPos.set(0, 0, formScaleRef.current * ZOOM_WAVE);
      ca.toTarget.set(0, 0, 0);
      ca.t      = 0;
      ca.active = true;
    }
  };

  const togglePlay = () => {
    if (!audioElRef.current) return;
    if (isPlaying) {
      audioElRef.current.pause();
      isPlayingRef.current = false;
      setIsPlaying(false);
      flyToSongPos(selectedIdxRef.current);
    } else {
      audioElRef.current.play();
      isPlayingRef.current = true;
      setIsPlaying(true);
      // Fly back to formation view if user navigated away
      const ca      = camAnimRef.current;
      const camera  = cameraRef.current;
      const controls = controlsRef.current;
      if (camera && controls) {
        ca.fromPos.copy(camera.position);
        ca.fromTarget.copy(controls.target);
        ca.toPos.set(0, 0, formScaleRef.current * ZOOM_WAVE);
        ca.toTarget.set(0, 0, 0);
        ca.t = 0; ca.active = true;
      }
    }
  };

  // ── Camera fly to a song's UMAP base position ────────────────────────────
  const selectedIdxRef   = useRef(null);
  const lastPlayedIdxRef = useRef(null); // persists after stop — drives the blink

  const flyToSongPos = (idx) => {
    const camera   = cameraRef.current;
    const controls = controlsRef.current;
    if (!camera || !controls || idx === null) return;
    const base = basePositionsRef.current[idx];
    if (!base) return;
    const ca = camAnimRef.current;
    ca.fromPos.copy(camera.position);
    ca.fromTarget.copy(controls.target);
    ca.toTarget.copy(base);
    // Sit in front of the point at a comfortable distance
    ca.toPos.set(base.x, base.y, base.z + formScaleRef.current * ZOOM_RETURN);
    ca.t = 0;
    ca.active = true;
  };

  // ── Raycasting ───────────────────────────────────────────────────────────
  const handleClick = (e) => {
    const mount  = mountRef.current;
    const camera = cameraRef.current;
    const mesh   = pointsRef.current;
    if (!mount || !camera || !mesh) return;

    const rect = mount.getBoundingClientRect();
    const nx = ((e.clientX - rect.left) / rect.width)  *  2 - 1;
    const ny = -((e.clientY - rect.top)  / rect.height) *  2 + 1;
    raycasterRef.current.setFromCamera(new THREE.Vector2(nx, ny), camera);
    const hits = raycasterRef.current.intersectObject(pointsRef.current);
    if (!hits.length) return;

    const idx = hits[0].index;
    const pt  = pointsDataRef.current[idx];
    if (!pt?.audio) return;

    selectedIdxRef.current   = idx;
    lastPlayedIdxRef.current = idx;
    setSelectedSong(pt);
    playAudioUrl(`${process.env.REACT_APP_API_BASE_URL || 'http://127.0.0.1:5000'}/audio/${pt.audio}`);
  };

  // ── Search ───────────────────────────────────────────────────────────────
  const filteredSongs = useMemo(() => {
    if (!searchQuery.trim()) return [];
    const q = searchQuery.toLowerCase();
    return embData
      .filter(e => e.name.toLowerCase().includes(q) || e.tag.toLowerCase().includes(q))
      .slice(0, 50);
  }, [searchQuery, embData]);

  // ── Render ───────────────────────────────────────────────────────────────
  const songColor = selectedSong
    ? getColorForGenre(selectedSong.tag, genreColorMap)
    : [0, 200, 255];
  const colorStr = `rgb(${songColor[0]},${songColor[1]},${songColor[2]})`;

  return (
    <div style={{ width: '100%', height: '100%', position: 'relative', background: '#000' }}>

      {/* Three.js canvas — always mounted */}
      <div ref={mountRef} style={{ width: '100%', height: '100%' }} onClick={handleClick} />

      {/* Loading overlay */}
      {loading && (
        <div style={{ ...S.overlay, ...S.center }}>
          <div style={S.spinner} />
          <p style={{ marginTop: 20, color: 'white' }}>Loading particles...</p>
        </div>
      )}

      {/* Error overlay */}
      {error && (
        <div style={{ ...S.overlay, ...S.center }}>
          <div style={{ fontSize: 42, marginBottom: 12 }}>⚠</div>
          <h2 style={{ color: 'white' }}>Error</h2>
          <p style={{ color: '#aaa' }}>{error}</p>
        </div>
      )}

      {/* Now-playing bar */}
      {selectedSong && (
        <div style={{ ...S.nowPlaying, borderColor: `${colorStr}66` }}>
          <div style={{ flex: 1, minWidth: 0, marginRight: 12 }}>
            <div style={{
              fontSize: 13, fontWeight: 600, color: 'white',
              whiteSpace: 'nowrap', overflow: 'hidden', textOverflow: 'ellipsis',
            }}>
              {selectedSong.name}
            </div>
            <div style={{ fontSize: 11, color: colorStr, marginTop: 2 }}>{selectedSong.tag}</div>
          </div>
          <div style={{ display: 'flex', alignItems: 'center', gap: 8, flex: 2 }}>
            <span style={S.timeLabel}>{fmt(currentTime)}</span>
            <input
              type="range" min={0} max={duration || 0} value={currentTime}
              onChange={e => {
                const t = +e.target.value;
                setCurrentTime(t);
                if (audioElRef.current) audioElRef.current.currentTime = t;
              }}
              style={{
                flex: 1, height: 3, cursor: 'pointer', accentColor: colorStr,
                background: `linear-gradient(to right,${colorStr} ${(currentTime / (duration || 1)) * 100}%,rgba(255,255,255,0.15) 0)`,
              }}
            />
            <span style={S.timeLabel}>{fmt(duration)}</span>
          </div>
          <div style={{ display: 'flex', gap: 8, marginLeft: 12 }}>
            <button onClick={togglePlay}
              style={{ ...S.btn, background: colorStr, color: '#000' }}>
              {isPlaying ? '⏸' : '▶'}
            </button>
            <button onClick={() => { stopAudio(); flyToSongPos(selectedIdxRef.current); setSelectedSong(null); }}
              style={{ ...S.btn, background: 'rgba(255,255,255,0.1)', color: 'white' }}>
              ✕
            </button>
          </div>
        </div>
      )}

      {/* Genre legend */}
      {genreColorMap && !loading && (
        <div style={S.legend}>
          <div style={{ fontWeight: 700, marginBottom: 8 }}>Genres</div>
          {Object.entries(genreColorMap).map(([g, c]) => (
            <div key={g} style={{ display: 'flex', alignItems: 'center', marginBottom: 4 }}>
              <div style={{
                width: 9, height: 9, borderRadius: 2, marginRight: 7, flexShrink: 0,
                background: `rgb(${c[0]},${c[1]},${c[2]})`,
              }} />
              <span>{g}</span>
            </div>
          ))}
        </div>
      )}

      {/* Search panel */}
      {!loading && (
        <div style={S.search}>
          <div style={{ fontWeight: 700, fontSize: 13, marginBottom: 8 }}>Search Songs</div>
          <input
            type="text" placeholder="Song name or genre..."
            value={searchQuery} onChange={e => setSearchQuery(e.target.value)}
            style={S.input}
          />
          {searchQuery && (
            <div style={{ fontSize: 10, opacity: 0.55, marginBottom: 6 }}>
              {filteredSongs.length} result{filteredSongs.length !== 1 ? 's' : ''}
            </div>
          )}
          {searchQuery && filteredSongs.length > 0 && (
            <div style={{ overflowY: 'auto', maxHeight: 240 }}>
              {filteredSongs.map((emb, idx) => {
                const c = getColorForGenre(emb.tag, genreColorMap);
                return (
                  <div key={idx}
                    onClick={() => {
                      const ptIdx = pointsDataRef.current.findIndex(p => p.name === emb.name);
                      if (ptIdx >= 0) {
                        lastPlayedIdxRef.current = ptIdx;
                        flyToSongPos(ptIdx);
                      }
                      setSearchQuery('');
                    }}
                    style={S.searchItem}
                    onMouseEnter={e => e.currentTarget.style.background = 'rgba(255,255,255,0.1)'}
                    onMouseLeave={e => e.currentTarget.style.background = 'rgba(255,255,255,0.04)'}
                  >
                    <div style={{
                      fontSize: 12, fontWeight: 500,
                      whiteSpace: 'nowrap', overflow: 'hidden', textOverflow: 'ellipsis',
                    }}>
                      {emb.name}
                    </div>
                    <div style={{ display: 'flex', alignItems: 'center', gap: 5, marginTop: 2 }}>
                      <span style={{
                        width: 6, height: 6, borderRadius: '50%', flexShrink: 0,
                        background: `rgb(${c[0]},${c[1]},${c[2]})`,
                      }} />
                      <span style={{ fontSize: 10, opacity: 0.6 }}>{emb.tag}</span>
                    </div>
                  </div>
                );
              })}
            </div>
          )}
          {searchQuery && !filteredSongs.length && (
            <div style={{ fontSize: 11, opacity: 0.45, padding: '10px 0', textAlign: 'center' }}>
              No results for "{searchQuery}"
            </div>
          )}
        </div>
      )}
    </div>
  );
};

// ── Styles ─────────────────────────────────────────────────────────────────
const S = {
  overlay:  { position: 'absolute', inset: 0, zIndex: 20 },
  center:   { display: 'flex', flexDirection: 'column', alignItems: 'center', justifyContent: 'center', background: '#000' },
  spinner:  { width: 44, height: 44, border: '4px solid rgba(255,255,255,0.15)', borderTop: '4px solid white', borderRadius: '50%', animation: 'spin 1s linear infinite' },
  nowPlaying: {
    position: 'absolute', bottom: 16, left: '50%', transform: 'translateX(-50%)',
    width: 'min(620px, 90vw)', background: 'rgba(0,0,0,0.88)',
    border: '1px solid', borderRadius: 10, padding: '11px 16px',
    display: 'flex', alignItems: 'center', gap: 8,
    zIndex: 10, backdropFilter: 'blur(10px)', boxShadow: '0 4px 30px rgba(0,0,0,0.7)',
    color: 'white',
  },
  timeLabel: { fontSize: 10, color: 'rgba(255,255,255,0.45)', fontFamily: 'monospace', minWidth: 30, textAlign: 'center' },
  btn: { width: 30, height: 30, borderRadius: '50%', border: 'none', cursor: 'pointer', fontSize: 12, display: 'flex', alignItems: 'center', justifyContent: 'center' },
  legend: { position: 'absolute', top: 14, right: 14, background: 'rgba(0,0,0,0.72)', color: 'white', padding: '11px 13px', borderRadius: 6, fontSize: 11, maxHeight: 'calc(100vh - 100px)', overflowY: 'auto', zIndex: 10 },
  search: { position: 'absolute', bottom: 80, left: 14, background: 'rgba(0,0,0,0.82)', color: 'white', padding: '11px 13px', borderRadius: 8, fontSize: 13, width: 270, maxHeight: 400, zIndex: 10, boxShadow: '0 4px 20px rgba(0,0,0,0.6)' },
  input: { width: '100%', padding: '7px 10px', background: 'rgba(255,255,255,0.07)', border: '1px solid rgba(255,255,255,0.2)', borderRadius: 4, color: 'white', fontSize: 12, outline: 'none', marginBottom: 7, boxSizing: 'border-box' },
  searchItem: { padding: '6px 8px', marginBottom: 3, background: 'rgba(255,255,255,0.04)', borderRadius: 4, cursor: 'pointer', transition: 'background 0.12s' },
};
