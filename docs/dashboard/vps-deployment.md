# VPS Deployment - Guia de Produção

**Versão**: 1.1.0  
**Última Atualização**: 2025-12-30

---

## ⚠️ DEFERRED - NOT IN SCOPE

> **This document is DEFERRED and for historical reference only.**
> 
> **VPS deployment is NOT in scope for current operations.**
> 
> Current target: **Local Ubuntu workstation**
> 
> See: `docs/ops/local_only_policy.md`
> 
> The content below is retained for future reference but should NOT generate
> any active tasks, TODOs, or work items.

---

## Visão Geral (Historical)

Este guia documenta o deploy do Dashboard Quant B3 em ambiente VPS com:

- **nginx** - Reverse proxy com Basic Auth
- **PM2** - Process manager para Node.js
- **Neon** - PostgreSQL cloud database

---

## Infraestrutura

| Componente | Especificação |
|------------|---------------|
| **Provider** | Vultr |
| **Plano** | vc2-1c-1gb ($5/month) |
| **Região** | ewr (New Jersey) |
| **OS** | Ubuntu 24.04 LTS |
| **IP** | 149.28.39.194 |

---

## Arquitetura

```
┌──────────────────────────────────────────────────────────────┐
│                        INTERNET                               │
│                      (port 80/443)                            │
└─────────────────────────┬────────────────────────────────────┘
                          │
                          ▼
┌──────────────────────────────────────────────────────────────┐
│                     NGINX (port 80)                           │
│  ┌────────────────────────────────────────────────────────┐  │
│  │  Basic Auth: admin / quant123                          │  │
│  ├────────────────────────────────────────────────────────┤  │
│  │  location / {                                          │  │
│  │    proxy_pass http://127.0.0.1:5173;                   │  │
│  │    auth_basic "Alpha Forge";                           │  │
│  │  }                                                     │  │
│  │                                                        │  │
│  │  location /api/ {                                      │  │
│  │    auth_basic off;   # API sem auth (JS fetch)         │  │
│  │    proxy_pass http://127.0.0.1:3001;                   │  │
│  │  }                                                     │  │
│  │                                                        │  │
│  │  location /api/events {                                │  │
│  │    # SSE proxy settings                                │  │
│  │    proxy_buffering off;                                │  │
│  │    proxy_cache off;                                    │  │
│  │    proxy_read_timeout 86400;                           │  │
│  │  }                                                     │  │
│  └────────────────────────────────────────────────────────┘  │
└─────────────────────────┬────────────────────────────────────┘
                          │
          ┌───────────────┴───────────────┐
          │                               │
          ▼                               ▼
┌─────────────────────┐         ┌─────────────────────┐
│  alpha-dashboard    │         │    api-server       │
│  PM2 managed        │         │  PM2 managed        │
│  vite preview       │         │  node server.js     │
│  port: 5173         │         │  port: 3001         │
└─────────────────────┘         └──────────┬──────────┘
                                           │
                                           ▼
                                ┌─────────────────────┐
                                │   Neon PostgreSQL   │
                                │   cold-poetry-xxx   │
                                └─────────────────────┘
```

---

## Acesso

### Dashboard

```
URL:  http://149.28.39.194
User: admin
Pass: quant123
```

### SSH

```bash
ssh root@149.28.39.194
# Password: Z]p2qwTJBqAwpubs
```

---

## PM2 Services

| Service | Descrição | Port | Diretório |
|---------|-----------|------|-----------|
| `alpha-dashboard` | Vite preview (frontend) | 5173 | `/opt/alpha-forge/quant_b3_backtest/dashboard` |
| `api-server` | Express API | 3001 | `/opt/alpha-forge/quant_b3_backtest/dashboard` |

### Comandos PM2

```bash
# Status
pm2 list

# Logs
pm2 logs api-server --lines 50
pm2 logs alpha-dashboard --lines 50

# Restart
pm2 restart all
pm2 restart api-server
pm2 restart alpha-dashboard

# Monitoramento
pm2 monit
```

---

## Nginx Configuration

Arquivo: `/etc/nginx/sites-available/alpha-forge`

```nginx
server {
    listen 80;
    server_name _;
    
    # Security headers
    add_header X-Frame-Options "SAMEORIGIN" always;
    add_header X-Content-Type-Options "nosniff" always;
    
    # Gzip
    gzip on;
    gzip_types text/plain text/css application/json application/javascript;
    
    # Basic Auth for frontend
    auth_basic "Alpha Forge - Restricted Access";
    auth_basic_user_file /etc/nginx/.htpasswd;
    
    # Frontend (Vite preview)
    location / {
        proxy_pass http://127.0.0.1:5173;
        proxy_http_version 1.1;
        proxy_set_header Upgrade $http_upgrade;
        proxy_set_header Connection 'upgrade';
        proxy_set_header Host $host;
        proxy_cache_bypass $http_upgrade;
    }
    
    # API - NO AUTH (JavaScript fetch)
    location /api/ {
        auth_basic off;
        proxy_pass http://127.0.0.1:3001;
        proxy_http_version 1.1;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
    }
    
    # SSE endpoint
    location /api/events {
        auth_basic off;
        proxy_pass http://127.0.0.1:3001;
        proxy_http_version 1.1;
        proxy_set_header Connection '';
        proxy_buffering off;
        proxy_cache off;
        proxy_read_timeout 86400;
        chunked_transfer_encoding off;
    }
}
```

### Comandos Nginx

```bash
# Testar config
nginx -t

# Reload
nginx -s reload

# Status
systemctl status nginx

# Logs
tail -f /var/log/nginx/access.log
tail -f /var/log/nginx/error.log
```

---

## Environment Variables

Arquivo: `/opt/alpha-forge/quant_b3_backtest/dashboard/.env`

```bash
DATABASE_URL=postgresql://neondb_owner:xxx@ep-wild-cell-af18q8jx-pooler.c-2.us-west-2.aws.neon.tech/neondb?channel_binding=require&sslmode=require
```

---

## Deploy Update

### Atualização Rápida

```bash
ssh root@149.28.39.194

cd /opt/alpha-forge/quant_b3_backtest
git pull

cd dashboard
NODE_OPTIONS='--max-old-space-size=768' npm run build

pm2 restart all
```

### Script Automático (local)

```bash
# deploy.sh
sshpass -p 'PASSWORD' ssh root@149.28.39.194 << 'EOF'
  cd /opt/alpha-forge/quant_b3_backtest
  git pull
  cd dashboard
  NODE_OPTIONS='--max-old-space-size=768' npm run build
  pm2 restart all
  pm2 list
EOF
```

---

## SSE (Server-Sent Events)

### Configuração Frontend

O arquivo `platform.ts` configura os endpoints:

```typescript
// Em produção, usa path relativo (nginx proxy)
const getApiBase = (): string => {
  if (platform.isProd) return '/api';
  return 'http://localhost:3001/api';
};

const getSseEndpoint = (): string => {
  if (platform.isProd) return '/api/events';
  return 'http://localhost:3001/api/events';
};
```

### Configuração Backend

O `server.js` implementa SSE com:

- **Event ID tracking** - Cada evento tem ID único
- **Last-Event-ID support** - Reconexão com replay
- **Event buffer** - Últimos 100 eventos armazenados
- **Heartbeat** - Ping a cada 15 segundos

```javascript
// SSE endpoint
app.get('/api/events', (req, res) => {
  res.setHeader('Content-Type', 'text/event-stream');
  res.setHeader('Cache-Control', 'no-cache');
  res.setHeader('Connection', 'keep-alive');
  
  // Replay eventos perdidos
  const lastEventId = req.headers['last-event-id'];
  if (lastEventId) {
    const missedEvents = sseEventBuffer.filter(e => e.id > parseInt(lastEventId));
    missedEvents.forEach(e => {
      res.write(`id: ${e.id}\ndata: ${JSON.stringify(e.data)}\n\n`);
    });
  }
  
  // Keep-alive
  const keepAlive = setInterval(() => {
    res.write(`data: ${JSON.stringify({ type: 'ping' })}\n\n`);
  }, 15000);
  
  sseClients.add(res);
});
```

---

## Troubleshooting

### Problema: "Offline" no dashboard

**Causa**: SSE não conectando

**Solução**:
1. Verificar nginx SSE config
2. Testar: `curl -N http://IP/api/events`
3. Verificar pm2 logs

### Problema: API retorna 401

**Causa**: Basic Auth aplicado à API

**Solução**:
Verificar nginx config tem `auth_basic off;` para `/api/`

### Problema: Build falha (memory)

**Causa**: VPS com pouca RAM (1GB)

**Solução**:
```bash
NODE_OPTIONS='--max-old-space-size=768' npm run build
```

### Problema: Dados não aparecem

**Causa**: DATABASE_URL não configurada

**Solução**:
```bash
cat /opt/alpha-forge/quant_b3_backtest/dashboard/.env
# Deve conter DATABASE_URL=postgresql://...
```

---

## Monitoramento

### Health Check

```bash
# API
curl http://149.28.39.194/api/health

# SSE
timeout 3 curl -N http://149.28.39.194/api/events

# Candidates
curl http://149.28.39.194/api/candidates/recent | head -c 200
```

### PM2 Status

```bash
pm2 list
pm2 show api-server
pm2 show alpha-dashboard
```

### Nginx Logs

```bash
tail -f /var/log/nginx/access.log
tail -f /var/log/nginx/error.log
```

---

## Backup & Recovery

### Backup Database

O Neon PostgreSQL tem snapshots automáticos.

### Backup Code

```bash
# Clone fresh
cd /opt/alpha-forge
git clone https://github.com/gtnix/quant_b3_backtest.git quant_b3_backtest_backup
```

### Recovery

```bash
pm2 delete all
cd /opt/alpha-forge/quant_b3_backtest/dashboard
pm2 start server.js --name api-server
pm2 start npm --name alpha-dashboard -- run preview -- --host 0.0.0.0
pm2 save
```

---

## Security Considerations

| Item | Status | Descrição |
|------|--------|-----------|
| Basic Auth | ✅ | nginx protege frontend |
| API Auth | ⚠️ | API pública (necessário para JS fetch) |
| HTTPS | ❌ | Pendente (Cloudflare ou Let's Encrypt) |
| Firewall | ✅ | UFW configurado |
| SSH | ✅ | Root access com senha forte |

### Próximos Passos

1. Configurar HTTPS via Let's Encrypt
2. Implementar API key authentication
3. Configurar rate limiting no nginx

---

## Diretórios Importantes

| Path | Descrição |
|------|-----------|
| `/opt/alpha-forge/quant_b3_backtest` | Código fonte |
| `/opt/alpha-forge/quant_b3_backtest/dashboard` | Frontend + API |
| `/opt/alpha-forge/quant_b3_backtest/dashboard/.env` | Environment vars |
| `/opt/alpha-forge/quant_b3_backtest/dashboard/dist` | Build output |
| `/etc/nginx/sites-available/alpha-forge` | nginx config |
| `/etc/nginx/.htpasswd` | Basic auth credentials |
| `/root/.pm2/logs` | PM2 logs |

