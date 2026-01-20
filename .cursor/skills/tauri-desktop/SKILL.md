---
name: tauri-desktop
description: Especialista em Tauri 2.0 para aplicações desktop de alta performance com React
---

# Tauri Desktop Expert

Especialista em desenvolvimento de aplicações desktop usando Tauri 2.0 com frontend React/TypeScript.

## Quando Usar

- Desenvolvimento de features desktop-específicas
- Problemas de comunicação Tauri <-> Frontend
- Otimização de performance desktop
- Build e deploy multiplataforma

## Regras Críticas

### 1. NUNCA usar URLs relativas para API

```typescript
// ERRADO - quebra no Tauri
fetch('/api/endpoint')

// CORRETO - usar URL absoluta
fetch('http://localhost:3001/api/endpoint')
```

### 2. Sempre verificar plataforma para features nativas

```typescript
import { platform } from '../lib/platform';

if (platform.isTauri) {
  const result = await invoke('command_name', { args });
} else {
  const result = await api.get('/fallback-endpoint');
}
```

### 3. Usar commands.ts para abstração

```typescript
import { cmd } from '../lib/commands';
const index = await cmd.loadIndex();
```

## Build

```bash
cd dashboard
CI=false cargo tauri build
```

## Problemas Comuns

1. **Asset not found** - URL relativa sendo tratada como arquivo local
2. **invoke error** - Comando Rust não existe
3. **API offline** - Server não iniciado (`node server.js`)

## Checklist Migração Web -> Desktop

- [ ] URLs de API são absolutas
- [ ] Features nativas têm fallback
- [ ] API server está rodando
