import React, { useState, useEffect, useRef } from "react";
import { Camera, Radar, Navigation, Target, Zap } from "lucide-react";

const GateNavigationSystem = () => {
  const canvasRef = useRef(null);
  const [systemState, setSystemState] = useState("INITIALIZING");
  const [detectedObjects, setDetectedObjects] = useState([]);
  const [validGates, setValidGates] = useState([]);
  const [targetGate, setTargetGate] = useState(null);
  const [colorRule, setColorRule] = useState(null);
  const [steeringAngle, setSteeringAngle] = useState(0);
  const [logs, setLogs] = useState([]);

  // 시뮬레이션 데이터 생성
  useEffect(() => {
    const interval = setInterval(() => {
      simulateNavigation();
    }, 2000);
    return () => clearInterval(interval);
  }, [colorRule, detectedObjects]);

  const addLog = (message, type = "info") => {
    setLogs((prev) => [
      ...prev.slice(-5),
      { msg: message, type, time: new Date().toLocaleTimeString() },
    ]);
  };

  const simulateNavigation = () => {
    // 시뮬레이션: 랜덤 객체 생성 (라이다 + 카메라)
    const mockObjects = generateMockObjects();
    setDetectedObjects(mockObjects);

    // 객체에 색상 힌트 부여
    const objectsWithColor = mockObjects.map((obj) => ({
      ...obj,
      color: classifyColorRegion(obj.hue),
    }));

    // 게이트 페어 찾기
    const gates = findGatePairs(objectsWithColor);
    setValidGates(gates);

    if (gates.length > 0) {
      // 첫 게이트로 색 규칙 학습
      if (!colorRule) {
        learnColorRule(gates[0]);
      }

      // 가장 가까운 게이트 선택
      const closest = gates.reduce((prev, curr) =>
        curr.distance < prev.distance ? curr : prev
      );
      setTargetGate(closest);

      // 조향각 계산
      const steering = closest.midAngle;
      setSteeringAngle(steering);

      setSystemState("NAVIGATING");
      addLog(
        `🎯 타겟: ${closest.distance.toFixed(1)}m, 조향: ${steering.toFixed(
          1
        )}°`,
        "success"
      );
    } else {
      setSystemState("SEARCHING");
      addLog("⚠️ 게이트 없음 - 탐색 모드", "warning");
    }

    // 캔버스 렌더링
    drawRadar();
  };

  const learnColorRule = (firstGate) => {
    const rule = {
      left: firstGate.left.color,
      right: firstGate.right.color,
    };
    setColorRule(rule);
    addLog(`🎓 색 규칙 학습: 왼쪽=${rule.left}, 오른쪽=${rule.right}`, "info");
  };

  const generateMockObjects = () => {
    const count = Math.floor(Math.random() * 4) + 2; // 2~5개 객체
    return Array.from({ length: count }, () => ({
      angle: (Math.random() - 0.5) * 80, // -40° ~ +40°
      distance: Math.random() * 2 + 1, // 1m ~ 3m
      hue: Math.random() < 0.5 ? Math.random() * 20 : Math.random() * 40 + 60, // 빨강 or 초록
    }));
  };

  const classifyColorRegion = (hue) => {
    // HSV Hue 기준: 0~20 = 빨강, 60~100 = 초록
    const BOUNDARY = 40;
    if (hue < BOUNDARY) {
      return "RED";
    } else {
      return "GREEN";
    }
  };

  const findGatePairs = (objects) => {
    const redObjs = objects.filter((o) => o.color === "RED");
    const greenObjs = objects.filter((o) => o.color === "GREEN");

    const pairs = [];
    for (const red of redObjs) {
      for (const green of greenObjs) {
        const angleDiff = Math.abs(red.angle - green.angle);

        // 게이트 조건: 15~50도 사이
        if (angleDiff > 15 && angleDiff < 50) {
          const left = red.angle < green.angle ? red : green;
          const right = red.angle < green.angle ? green : red;

          pairs.push({
            left,
            right,
            midAngle: (red.angle + green.angle) / 2,
            distance: (red.distance + green.distance) / 2,
          });
        }
      }
    }
    return pairs;
  };

  const drawRadar = () => {
    const canvas = canvasRef.current;
    if (!canvas) return;

    const ctx = canvas.getContext("2d");
    const width = canvas.width;
    const height = canvas.height;
    const centerX = width / 2;
    const centerY = height - 20;

    // 배경
    ctx.fillStyle = "#0a1929";
    ctx.fillRect(0, 0, width, height);

    // 레이더 격자
    ctx.strokeStyle = "#1e3a5f";
    ctx.lineWidth = 1;
    for (let i = 1; i <= 3; i++) {
      ctx.beginPath();
      ctx.arc(centerX, centerY, i * 80, Math.PI, 0);
      ctx.stroke();
    }

    // 각도 선
    for (let angle = -60; angle <= 60; angle += 30) {
      const rad = (angle * Math.PI) / 180;
      ctx.beginPath();
      ctx.moveTo(centerX, centerY);
      ctx.lineTo(centerX + Math.sin(rad) * 240, centerY - Math.cos(rad) * 240);
      ctx.stroke();
    }

    // FOV 표시 (87도)
    ctx.strokeStyle = "#fbbf24";
    ctx.lineWidth = 2;
    const fovRad = ((87 / 2) * Math.PI) / 180;
    ctx.beginPath();
    ctx.moveTo(centerX, centerY);
    ctx.lineTo(
      centerX - Math.sin(fovRad) * 240,
      centerY - Math.cos(fovRad) * 240
    );
    ctx.moveTo(centerX, centerY);
    ctx.lineTo(
      centerX + Math.sin(fovRad) * 240,
      centerY - Math.cos(fovRad) * 240
    );
    ctx.stroke();

    // 객체 그리기
    detectedObjects.forEach((obj) => {
      const rad = (obj.angle * Math.PI) / 180;
      const x = centerX + Math.sin(rad) * obj.distance * 80;
      const y = centerY - Math.cos(rad) * obj.distance * 80;

      ctx.fillStyle = obj.color === "RED" ? "#ef4444" : "#22c55e";
      ctx.beginPath();
      ctx.arc(x, y, 8, 0, Math.PI * 2);
      ctx.fill();

      // 거리 텍스트
      ctx.fillStyle = "#fff";
      ctx.font = "10px monospace";
      ctx.fillText(`${obj.distance.toFixed(1)}m`, x + 10, y);
    });

    // 타겟 게이트 표시
    if (targetGate) {
      const midRad = (targetGate.midAngle * Math.PI) / 180;
      const midX = centerX + Math.sin(midRad) * targetGate.distance * 80;
      const midY = centerY - Math.cos(midRad) * targetGate.distance * 80;

      ctx.strokeStyle = "#fbbf24";
      ctx.lineWidth = 3;
      ctx.beginPath();
      ctx.arc(midX, midY, 15, 0, Math.PI * 2);
      ctx.stroke();

      // 경로 선
      ctx.strokeStyle = "#3b82f6";
      ctx.lineWidth = 2;
      ctx.beginPath();
      ctx.moveTo(centerX, centerY);
      ctx.lineTo(midX, midY);
      ctx.stroke();
    }

    // 로봇 위치
    ctx.fillStyle = "#3b82f6";
    ctx.beginPath();
    ctx.arc(centerX, centerY, 10, 0, Math.PI * 2);
    ctx.fill();
  };

  return (
    <div className="w-full h-screen bg-gray-900 text-white p-4">
      {/* 헤더 */}
      <div className="flex justify-between items-center mb-4">
        <h1 className="text-2xl font-bold flex items-center gap-2">
          <Navigation className="w-6 h-6" />
          해양 로봇 게이트 항법 시스템
        </h1>
        <div className="flex gap-2">
          <div
            className={`px-3 py-1 rounded ${
              systemState === "NAVIGATING"
                ? "bg-green-600"
                : systemState === "SEARCHING"
                ? "bg-yellow-600"
                : "bg-gray-600"
            }`}
          >
            {systemState}
          </div>
        </div>
      </div>

      <div className="grid grid-cols-3 gap-4">
        {/* 왼쪽: 레이더 디스플레이 */}
        <div className="col-span-2 bg-gray-800 rounded-lg p-4">
          <div className="flex items-center gap-2 mb-2">
            <Radar className="w-5 h-5" />
            <span className="font-bold">통합 센서 뷰 (LiDAR + Camera)</span>
          </div>
          <canvas
            ref={canvasRef}
            width={600}
            height={400}
            className="w-full border border-gray-700 rounded"
          />
          <div className="mt-2 text-sm text-gray-400">
            <span className="text-yellow-400">■</span> Camera FOV: 87° |
            <span className="text-red-400"> ●</span> Red Region |
            <span className="text-green-400"> ●</span> Green Region
          </div>
        </div>

        {/* 오른쪽: 시스템 정보 */}
        <div className="space-y-4">
          {/* 센서 상태 */}
          <div className="bg-gray-800 rounded-lg p-4">
            <h3 className="font-bold mb-2 flex items-center gap-2">
              <Camera className="w-4 h-4" />
              센서 상태
            </h3>
            <div className="space-y-1 text-sm">
              <div className="flex justify-between">
                <span>LiDAR:</span>
                <span className="text-green-400">Active</span>
              </div>
              <div className="flex justify-between">
                <span>RealSense D435i:</span>
                <span className="text-green-400">Active</span>
              </div>
              <div className="flex justify-between">
                <span>감지 객체:</span>
                <span className="text-blue-400">
                  {detectedObjects.length}개
                </span>
              </div>
            </div>
          </div>

          {/* 색 규칙 */}
          {colorRule && (
            <div className="bg-gray-800 rounded-lg p-4">
              <h3 className="font-bold mb-2 flex items-center gap-2">
                <Zap className="w-4 h-4" />
                학습된 색 규칙
              </h3>
              <div className="space-y-1 text-sm">
                <div className="flex justify-between items-center">
                  <span>왼쪽 라인:</span>
                  <span
                    className={`px-2 py-1 rounded ${
                      colorRule.left === "RED" ? "bg-red-600" : "bg-green-600"
                    }`}
                  >
                    {colorRule.left}
                  </span>
                </div>
                <div className="flex justify-between items-center">
                  <span>오른쪽 라인:</span>
                  <span
                    className={`px-2 py-1 rounded ${
                      colorRule.right === "RED" ? "bg-red-600" : "bg-green-600"
                    }`}
                  >
                    {colorRule.right}
                  </span>
                </div>
              </div>
            </div>
          )}

          {/* 타겟 정보 */}
          {targetGate && (
            <div className="bg-gray-800 rounded-lg p-4">
              <h3 className="font-bold mb-2 flex items-center gap-2">
                <Target className="w-4 h-4" />
                현재 타겟
              </h3>
              <div className="space-y-1 text-sm">
                <div className="flex justify-between">
                  <span>거리:</span>
                  <span className="text-blue-400">
                    {targetGate.distance.toFixed(2)}m
                  </span>
                </div>
                <div className="flex justify-between">
                  <span>방위각:</span>
                  <span className="text-blue-400">
                    {targetGate.midAngle.toFixed(1)}°
                  </span>
                </div>
                <div className="flex justify-between">
                  <span>조향각:</span>
                  <span className="text-yellow-400">
                    {steeringAngle.toFixed(1)}°
                  </span>
                </div>
              </div>
            </div>
          )}

          {/* 로그 */}
          <div className="bg-gray-800 rounded-lg p-4">
            <h3 className="font-bold mb-2">시스템 로그</h3>
            <div className="space-y-1 text-xs font-mono">
              {logs.map((log, i) => (
                <div
                  key={i}
                  className={`${
                    log.type === "success"
                      ? "text-green-400"
                      : log.type === "warning"
                      ? "text-yellow-400"
                      : "text-gray-400"
                  }`}
                >
                  [{log.time}] {log.msg}
                </div>
              ))}
            </div>
          </div>
        </div>
      </div>

      {/* 하단: 검출된 게이트 리스트 */}
      <div className="mt-4 bg-gray-800 rounded-lg p-4">
        <h3 className="font-bold mb-2">
          검출된 게이트 ({validGates.length}개)
        </h3>
        <div className="grid grid-cols-4 gap-2">
          {validGates.map((gate, i) => (
            <div
              key={i}
              className={`p-2 rounded border ${
                targetGate === gate
                  ? "border-yellow-400 bg-gray-700"
                  : "border-gray-600"
              }`}
            >
              <div className="text-xs">
                <div>Gate #{i + 1}</div>
                <div className="text-gray-400">
                  Left:{" "}
                  <span
                    className={
                      gate.left.color === "RED"
                        ? "text-red-400"
                        : "text-green-400"
                    }
                  >
                    {gate.left.color}
                  </span>
                </div>
                <div className="text-gray-400">
                  Right:{" "}
                  <span
                    className={
                      gate.right.color === "RED"
                        ? "text-red-400"
                        : "text-green-400"
                    }
                  >
                    {gate.right.color}
                  </span>
                </div>
                <div className="text-blue-400">{gate.distance.toFixed(1)}m</div>
              </div>
            </div>
          ))}
        </div>
      </div>
    </div>
  );
};

export default GateNavigationSystem;
