/**
 * Three.js 3D Background Scene — Particles, Neural Network, Code Rain
 * Uses React Three Fiber with post-processing
 */
import { useRef, useMemo, useCallback } from "react";
import { Canvas, useFrame } from "@react-three/fiber";
import { OrbitControls, Float, Line } from "@react-three/drei";
import {
  EffectComposer,
  Bloom,
  ChromaticAberration,
} from "@react-three/postprocessing";
import * as THREE from "three";
import { BlendFunction } from "postprocessing";
import { useStore } from "../../store/useStore";

/* ───── Floating Particles ───── */
function Particles({ count = 1500 }: { count?: number }) {
  const mesh = useRef<THREE.Points>(null!);
  const { positions, colors, speeds } = useMemo(() => {
    const positions = new Float32Array(count * 3);
    const colors = new Float32Array(count * 3);
    const speeds = new Float32Array(count);
    const cyan = new THREE.Color("#00f7ff");
    const magenta = new THREE.Color("#ff00aa");
    const purple = new THREE.Color("#8b5cf6");
    const palette = [cyan, magenta, purple];
    for (let i = 0; i < count; i++) {
      positions[i * 3] = (Math.random() - 0.5) * 40;
      positions[i * 3 + 1] = (Math.random() - 0.5) * 40;
      positions[i * 3 + 2] = (Math.random() - 0.5) * 40;
      const c = palette[Math.floor(Math.random() * palette.length)];
      colors[i * 3] = c.r;
      colors[i * 3 + 1] = c.g;
      colors[i * 3 + 2] = c.b;
      speeds[i] = 0.2 + Math.random() * 0.8;
    }
    return { positions, colors, speeds };
  }, [count]);

  useFrame(({ clock }) => {
    const t = clock.getElapsedTime();
    const posAttr = mesh.current.geometry.attributes
      .position as THREE.BufferAttribute;
    const arr = posAttr.array as Float32Array;
    for (let i = 0; i < count; i++) {
      arr[i * 3 + 1] += Math.sin(t * speeds[i] + i) * 0.003;
      arr[i * 3] += Math.cos(t * speeds[i] * 0.5 + i) * 0.002;
    }
    posAttr.needsUpdate = true;
  });

  return (
    <points ref={mesh}>
      <bufferGeometry>
        <bufferAttribute
          attach="attributes-position"
          count={count}
          array={positions}
          itemSize={3}
        />
        <bufferAttribute
          attach="attributes-color"
          count={count}
          array={colors}
          itemSize={3}
        />
      </bufferGeometry>
      <pointsMaterial
        size={0.06}
        vertexColors
        transparent
        opacity={0.8}
        sizeAttenuation
        blending={THREE.AdditiveBlending}
        depthWrite={false}
      />
    </points>
  );
}

/* ───── Neural Network Nodes ───── */
function NeuralNetwork({ nodeCount = 40 }: { nodeCount?: number }) {
  const groupRef = useRef<THREE.Group>(null!);
  const isQuerying = useStore((s) => s.isQuerying);

  const { nodes, edges } = useMemo(() => {
    const nodes: THREE.Vector3[] = [];
    for (let i = 0; i < nodeCount; i++) {
      nodes.push(
        new THREE.Vector3(
          (Math.random() - 0.5) * 20,
          (Math.random() - 0.5) * 15,
          (Math.random() - 0.5) * 15,
        ),
      );
    }
    const edges: [number, number][] = [];
    for (let i = 0; i < nodeCount; i++) {
      const closest: { idx: number; dist: number }[] = [];
      for (let j = 0; j < nodeCount; j++) {
        if (i === j) continue;
        closest.push({ idx: j, dist: nodes[i].distanceTo(nodes[j]) });
      }
      closest.sort((a, b) => a.dist - b.dist);
      for (let k = 0; k < Math.min(2, closest.length); k++) {
        if (closest[k].dist < 10) edges.push([i, closest[k].idx]);
      }
    }
    return { nodes, edges };
  }, [nodeCount]);

  useFrame(({ clock }) => {
    const t = clock.getElapsedTime();
    groupRef.current.rotation.y = t * 0.02;
    groupRef.current.rotation.x = Math.sin(t * 0.01) * 0.1;
  });

  return (
    <group ref={groupRef}>
      {nodes.map((pos, i) => (
        <mesh key={i} position={pos}>
          <sphereGeometry args={[0.08, 8, 8]} />
          <meshBasicMaterial
            color={isQuerying ? "#ff00aa" : "#00f7ff"}
            transparent
            opacity={0.7 + Math.sin(i) * 0.3}
          />
        </mesh>
      ))}
      {edges.map(([a, b], i) => (
        <Line
          key={`e${i}`}
          points={[nodes[a], nodes[b]]}
          color="#00f7ff"
          transparent
          opacity={0.12}
          lineWidth={1}
        />
      ))}
    </group>
  );
}

/* ───── Code Rain (Matrix-style) ───── */
function CodeRain({ columns = 30 }: { columns?: number }) {
  const groupRef = useRef<THREE.Group>(null!);
  const chars =
    "ABCDEF0123456789{}[]();=>import const function class return async await".split(
      "",
    );

  const rainData = useMemo(() => {
    return Array.from({ length: columns }, (_, i) => ({
      x: (i - columns / 2) * 0.8 + (Math.random() - 0.5) * 0.3,
      z: -8 - Math.random() * 10,
      speed: 0.5 + Math.random() * 1.5,
      offset: Math.random() * 20,
      char: chars[Math.floor(Math.random() * chars.length)],
    }));
  }, [columns]);

  useFrame(({ clock }) => {
    const t = clock.getElapsedTime();
    const children = groupRef.current.children;
    for (let i = 0; i < children.length; i++) {
      const data = rainData[i];
      const y = ((t * data.speed + data.offset) % 20) - 10;
      children[i].position.y = -y;
    }
  });

  return (
    <group ref={groupRef}>
      {rainData.map((d, i) => (
        <mesh key={i} position={[d.x, 0, d.z]}>
          <planeGeometry args={[0.3, 0.3]} />
          <meshBasicMaterial
            color="#00f7ff"
            transparent
            opacity={0.15}
            side={THREE.DoubleSide}
          />
        </mesh>
      ))}
    </group>
  );
}

/* ───── Wireframe Structures ───── */
function WireframeStructures() {
  const ref1 = useRef<THREE.Mesh>(null!);
  const ref2 = useRef<THREE.Mesh>(null!);

  useFrame(({ clock }) => {
    const t = clock.getElapsedTime();
    ref1.current.rotation.x = t * 0.1;
    ref1.current.rotation.z = t * 0.05;
    ref2.current.rotation.y = t * 0.08;
    ref2.current.rotation.x = Math.sin(t * 0.1) * 0.3;
  });

  return (
    <>
      <mesh ref={ref1} position={[-12, 3, -8]}>
        <icosahedronGeometry args={[2, 1]} />
        <meshBasicMaterial
          color="#8b5cf6"
          wireframe
          transparent
          opacity={0.15}
        />
      </mesh>
      <mesh ref={ref2} position={[12, -2, -6]}>
        <torusGeometry args={[1.5, 0.4, 8, 12]} />
        <meshBasicMaterial
          color="#ff00aa"
          wireframe
          transparent
          opacity={0.12}
        />
      </mesh>
    </>
  );
}

/* ───── Grid Floor ───── */
function GridFloor() {
  return (
    <gridHelper args={[60, 60, "#0a2a3a", "#0a1a2a"]} position={[0, -10, 0]} />
  );
}

/* ───── Main Scene ───── */
export default function Scene() {
  return (
    <div
      style={{
        position: "fixed",
        inset: 0,
        zIndex: 0,
        pointerEvents: "none",
      }}
    >
      <Canvas
        camera={{ position: [0, 0, 15], fov: 60, near: 0.1, far: 100 }}
        gl={{ antialias: true, alpha: true }}
        style={{ background: "transparent" }}
      >
        <ambientLight intensity={0.3} />
        <Particles count={1500} />
        <NeuralNetwork nodeCount={40} />
        <CodeRain columns={30} />
        <WireframeStructures />
        <GridFloor />
        <Float speed={0.5} rotationIntensity={0.1} floatIntensity={0.3}>
          <mesh position={[0, 0, -15]}>
            <torusKnotGeometry args={[3, 0.8, 100, 16]} />
            <meshBasicMaterial
              color="#8b5cf6"
              wireframe
              transparent
              opacity={0.06}
            />
          </mesh>
        </Float>
        <EffectComposer>
          <Bloom
            intensity={0.5}
            luminanceThreshold={0.1}
            luminanceSmoothing={0.9}
            mipmapBlur
          />
          {/* @ts-ignore - offset type mismatch between versions */}
          <ChromaticAberration
            blendFunction={BlendFunction.NORMAL}
            offset={new THREE.Vector2(0.0005, 0.0005) as any}
          />
        </EffectComposer>
      </Canvas>
    </div>
  );
}
