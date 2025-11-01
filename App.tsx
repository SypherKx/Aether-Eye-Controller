import React, { useCallback, useEffect, useMemo, useRef, useState } from 'react';
import { SafeAreaView, View, Text, TextInput, TouchableOpacity, StyleSheet, ScrollView } from 'react-native';
import * as Speech from 'expo-speech';

function Button({ title, onPress, active }: { title: string; onPress?: () => void; active?: boolean }) {
  return (
    <TouchableOpacity onPress={onPress} style={[styles.button, active ? styles.buttonActive : undefined]}>
      <Text style={[styles.buttonText, active ? styles.buttonTextActive : undefined]}>{title}</Text>
    </TouchableOpacity>
  );
}

export default function App() {
  const [serverIp, setServerIp] = useState('');
  const [esp32Ip, setEsp32Ip] = useState('');
  const [connected, setConnected] = useState(false);
  const [flashMode, setFlashMode] = useState<'auto' | 'on' | 'off'>('auto');
  const [motion, setMotion] = useState(false);
  const [scanRunning, setScanRunning] = useState(false);
  const [resultText, setResultText] = useState('');
  const wsRef = useRef<WebSocket | null>(null);

  const baseUrl = useMemo(() => (serverIp ? `http://${serverIp}:8000` : ''), [serverIp]);

  const speak = useCallback((text: string) => {
    if (!text) return;
    Speech.stop();
    Speech.speak(text, { language: 'en' });
  }, []);

  const connect = useCallback(async () => {
    if (!baseUrl || !esp32Ip) return;
    try {
      const r = await fetch(`${baseUrl}/connect`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ ip: esp32Ip })
      });
      const data = await r.json();
      if (data.ok) {
        setConnected(true);
        // open websocket
        const wsUrl = baseUrl.replace(/^http/, 'ws') + '/ws';
        const ws = new WebSocket(wsUrl);
        ws.onopen = () => {
          console.log('WebSocket connected');
        };
        ws.onmessage = (ev) => {
          try {
            const msg = JSON.parse(ev.data);
            if (msg.type === 'scan_started') {
              setScanRunning(true);
            } else if (msg.type === 'scan_complete') {
              setScanRunning(false);
              if (msg.summary) {
                setResultText(msg.summary);
                // Auto speak the summary when it arrives
                speak(String(msg.summary));
              }
            } else if (msg.type === 'motion') {
              // Optional: show transient motion message
            }
          } catch (e) {
            console.warn('Failed to parse WebSocket message:', e);
          }
        };
        ws.onclose = () => {
          console.log('WebSocket closed');
          wsRef.current = null;
        };
        ws.onerror = (error) => {
          console.error('WebSocket error:', error);
          setConnected(false);
        };
        wsRef.current = ws;
      }
    } catch (e) {
      setConnected(false);
    }
  }, [baseUrl, esp32Ip, speak]);

  const setFlash = useCallback(async (mode: 'auto' | 'on' | 'off') => {
    if (!baseUrl) return;
    setFlashMode(mode);
    try {
      await fetch(`${baseUrl}/flashlight`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ mode })
      });
    } catch {}
  }, [baseUrl]);

  const toggleMotion = useCallback(async () => {
    if (!baseUrl) return;
    const enabled = !motion;
    setMotion(enabled);
    try {
      await fetch(`${baseUrl}/motion`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ enabled })
      });
    } catch {}
  }, [baseUrl, motion]);

  const toggleScan = useCallback(async () => {
    if (!baseUrl) return;
    if (scanRunning) return; // stop not implemented
    setScanRunning(true);
    try {
      await fetch(`${baseUrl}/scan`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ start: true })
      });
    } catch {
      setScanRunning(false);
    }
  }, [baseUrl, scanRunning]);

  useEffect(() => {
    return () => {
      if (wsRef.current) {
        wsRef.current.close();
        wsRef.current = null;
      }
    };
  }, []);

  return (
    <SafeAreaView style={styles.container}>
      <ScrollView contentContainerStyle={styles.scroll}>
        <Text style={styles.title}>Aether YOLO Controller</Text>

        <View style={styles.card}>
          <Text style={styles.cardTitle}>Connection</Text>
          <TextInput
            placeholder="Server IP (PC) e.g. 192.168.1.100"
            value={serverIp}
            onChangeText={setServerIp}
            style={styles.input}
            autoCapitalize="none"
          />
          <TextInput
            placeholder="ESP32 IP e.g. 192.168.1.50"
            value={esp32Ip}
            onChangeText={setEsp32Ip}
            style={styles.input}
            autoCapitalize="none"
          />
          <Button title={connected ? 'Connected' : 'Connect'} onPress={connect} active={connected} />
        </View>

        <View style={styles.card}>
          <Text style={styles.cardTitle}>Flashlight</Text>
          <View style={styles.row}>
            <Button title="Auto" active={flashMode === 'auto'} onPress={() => setFlash('auto')} />
            <Button title="On" active={flashMode === 'on'} onPress={() => setFlash('on')} />
            <Button title="Off" active={flashMode === 'off'} onPress={() => setFlash('off')} />
          </View>
        </View>

        <View style={styles.card}>
          <Text style={styles.cardTitle}>Motion</Text>
          <Button title={motion ? 'Turn Off' : 'Turn On'} active={motion} onPress={toggleMotion} />
        </View>

        <View style={styles.card}>
          <Text style={styles.cardTitle}>Room Scan</Text>
          <Button title={scanRunning ? 'Running…' : 'Start Scan'} active={scanRunning} onPress={toggleScan} />
        </View>

        <View style={styles.card}>
          <Text style={styles.cardTitle}>Result</Text>
          <Text style={styles.result}>{resultText || 'No result yet.'}</Text>
          <Button title="Speak" onPress={() => speak(resultText)} />
        </View>
      </ScrollView>
    </SafeAreaView>
  );
}

const styles = StyleSheet.create({
  container: { flex: 1, backgroundColor: '#0b0f14' },
  scroll: { padding: 16 },
  title: { color: 'white', fontSize: 22, fontWeight: '700', marginBottom: 16 },
  card: { backgroundColor: '#121826', padding: 16, borderRadius: 12, marginBottom: 12, borderWidth: 1, borderColor: '#1f2937' },
  cardTitle: { color: '#cbd5e1', marginBottom: 8, fontWeight: '600' },
  input: { backgroundColor: '#0b0f14', color: 'white', padding: 10, borderRadius: 8, borderWidth: 1, borderColor: '#1f2937', marginBottom: 8 },
  button: { backgroundColor: '#111827', borderWidth: 1, borderColor: '#374151', paddingVertical: 10, paddingHorizontal: 14, borderRadius: 10, marginRight: 8, marginTop: 6 },
  buttonActive: { backgroundColor: '#2563eb', borderColor: '#2563eb' },
  buttonText: { color: '#cbd5e1', fontWeight: '600' },
  buttonTextActive: { color: 'white' },
  row: { flexDirection: 'row', alignItems: 'center' },
  result: { color: 'white', marginBottom: 8 }
});
