import { useRef, useMemo } from 'react';
import { Canvas, useFrame } from '@react-three/fiber';
import { OrbitControls, Text, Html } from '@react-three/drei';
import * as THREE from 'three';

interface ParetoChartProps {
  data: Array<{
    id: string;
    cagr: number;
    sharpe: number;
    maxDrawdown: number;
    paretoRank: number;
  }>;
}

function DataPoint({ 
  position, 
  color, 
  id, 
  isPareto 
}: { 
  position: [number, number, number]; 
  color: string; 
  id: string; 
  isPareto: boolean;
}) {
  const meshRef = useRef<THREE.Mesh>(null);
  
  useFrame((state) => {
    if (meshRef.current && isPareto) {
      meshRef.current.scale.setScalar(1 + Math.sin(state.clock.elapsedTime * 2) * 0.1);
    }
  });

  return (
    <mesh ref={meshRef} position={position}>
      <sphereGeometry args={[isPareto ? 0.08 : 0.05, 16, 16]} />
      <meshStandardMaterial 
        color={color} 
        emissive={color}
        emissiveIntensity={isPareto ? 0.5 : 0.2}
        transparent
        opacity={isPareto ? 1 : 0.7}
      />
      {isPareto && (
        <Html position={[0.15, 0.1, 0]} className="pointer-events-none">
          <div className="text-[10px] font-mono text-profit whitespace-nowrap bg-terminal-bg/80 px-1 rounded">
            {id.substring(0, 8)}
          </div>
        </Html>
      )}
    </mesh>
  );
}

function Axes() {
  return (
    <group>
      {/* X axis - CAGR */}
      <line>
        <bufferGeometry>
          <bufferAttribute
            attach="attributes-position"
            count={2}
            array={new Float32Array([0, 0, 0, 2, 0, 0])}
            itemSize={3}
          />
        </bufferGeometry>
        <lineBasicMaterial color="#00ff88" opacity={0.5} transparent />
      </line>
      <Text position={[2.2, 0, 0]} fontSize={0.12} color="#00ff88">
        CAGR
      </Text>

      {/* Y axis - Sharpe */}
      <line>
        <bufferGeometry>
          <bufferAttribute
            attach="attributes-position"
            count={2}
            array={new Float32Array([0, 0, 0, 0, 2, 0])}
            itemSize={3}
          />
        </bufferGeometry>
        <lineBasicMaterial color="#00d4ff" opacity={0.5} transparent />
      </line>
      <Text position={[0, 2.2, 0]} fontSize={0.12} color="#00d4ff">
        Sharpe
      </Text>

      {/* Z axis - MaxDD */}
      <line>
        <bufferGeometry>
          <bufferAttribute
            attach="attributes-position"
            count={2}
            array={new Float32Array([0, 0, 0, 0, 0, 2])}
            itemSize={3}
          />
        </bufferGeometry>
        <lineBasicMaterial color="#8b5cf6" opacity={0.5} transparent />
      </line>
      <Text position={[0, 0, 2.2]} fontSize={0.12} color="#8b5cf6">
        -MaxDD
      </Text>

      {/* Grid on XY plane */}
      <gridHelper args={[4, 10, '#1e1e2e', '#1e1e2e']} rotation={[Math.PI / 2, 0, 0]} />
    </group>
  );
}

function Scene({ data }: { data: ParetoChartProps['data'] }) {
  const points = useMemo(() => {
    // Normalize data to 0-2 range
    const maxCagr = Math.max(...data.map(d => d.cagr));
    const maxSharpe = Math.max(...data.map(d => d.sharpe));
    const maxDD = Math.max(...data.map(d => Math.abs(d.maxDrawdown)));

    return data.map(d => ({
      ...d,
      position: [
        (d.cagr / maxCagr) * 2,
        (d.sharpe / maxSharpe) * 2,
        (Math.abs(d.maxDrawdown) / maxDD) * 2,
      ] as [number, number, number],
      color: d.paretoRank === 0 ? '#00ff88' : d.paretoRank === 1 ? '#00d4ff' : '#3a3a4a',
      isPareto: d.paretoRank === 0,
    }));
  }, [data]);

  return (
    <>
      <ambientLight intensity={0.5} />
      <pointLight position={[10, 10, 10]} intensity={1} />
      <Axes />
      {points.map((point) => (
        <DataPoint
          key={point.id}
          position={point.position}
          color={point.color}
          id={point.id}
          isPareto={point.isPareto}
        />
      ))}
      <OrbitControls 
        enablePan={true}
        enableZoom={true}
        enableRotate={true}
        autoRotate={true}
        autoRotateSpeed={0.5}
      />
    </>
  );
}

export function ParetoChart({ data }: ParetoChartProps) {
  return (
    <div className="w-full h-full relative">
      <Canvas
        camera={{ position: [3, 3, 3], fov: 50 }}
        style={{ background: 'transparent' }}
      >
        <Scene data={data} />
      </Canvas>
      {/* Legend */}
      <div className="absolute bottom-2 left-2 flex items-center gap-4 text-xs bg-terminal-bg/80 px-2 py-1 rounded">
        <div className="flex items-center gap-1">
          <div className="w-2 h-2 rounded-full bg-profit" />
          <span className="text-terminal-muted">Pareto Front</span>
        </div>
        <div className="flex items-center gap-1">
          <div className="w-2 h-2 rounded-full bg-accent-cyan" />
          <span className="text-terminal-muted">Rank 1</span>
        </div>
        <div className="flex items-center gap-1">
          <div className="w-2 h-2 rounded-full bg-terminal-muted" />
          <span className="text-terminal-muted">Others</span>
        </div>
      </div>
    </div>
  );
}


























