#!/bin/bash
#############################################################################
# ALPHA FORGE - VPS Setup Script
# Professional backtest server with Nginx, PM2, and HTTP Basic Auth
#############################################################################

set -e

# Configuration
SWAP_SIZE="2G"
PROJECT_PATH="/opt/alpha-forge/quant_b3_backtest"
AUTH_USER="admin"
AUTH_PASS=$(openssl rand -base64 12 | tr -d '/+=' | head -c 16)

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
NC='\033[0m'

log() { echo -e "${GREEN}>>>${NC} $1"; }
step() { echo -e "\n${BLUE}[$1/9]${NC} ${CYAN}$2${NC}"; }

echo ""
echo -e "${BLUE}╔═══════════════════════════════════════════════════════════════════╗${NC}"
echo -e "${BLUE}║           ALPHA FORGE - SCG Server Setup                          ║${NC}"
echo -e "${BLUE}║           Professional Backtest Infrastructure                    ║${NC}"
echo -e "${BLUE}╚═══════════════════════════════════════════════════════════════════╝${NC}"
echo ""

# Step 1: Create Swap
step "1" "Configuring $SWAP_SIZE swap for Rust compilation..."
if [ ! -f /swapfile ]; then
    fallocate -l $SWAP_SIZE /swapfile
    chmod 600 /swapfile
    mkswap /swapfile
    swapon /swapfile
    echo '/swapfile none swap sw 0 0' >> /etc/fstab
    log "Swap created and enabled"
else
    log "Swap already exists"
fi
free -h | grep -i swap

# Step 2: Update system
step "2" "Updating system packages..."
apt update && apt upgrade -y

# Step 3: Install essential tools
step "3" "Installing essential tools..."
apt install -y \
    build-essential \
    pkg-config \
    libssl-dev \
    git \
    curl \
    wget \
    htop \
    tmux \
    jq \
    unzip \
    ca-certificates \
    gnupg \
    nginx \
    apache2-utils \
    certbot \
    python3-certbot-nginx \
    ufw

# Step 4: Install Rust
step "4" "Installing Rust..."
if ! command -v rustc &> /dev/null; then
    curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y
    source "$HOME/.cargo/env"
fi
rustc --version
cargo --version

# Step 5: Install Node.js 20 LTS
step "5" "Installing Node.js 20 LTS..."
if ! command -v node &> /dev/null; then
    curl -fsSL https://deb.nodesource.com/setup_20.x | bash -
    apt install -y nodejs
fi
node --version
npm --version

# Install PM2 globally
npm install -g pm2

# Step 6: Create project structure
step "6" "Creating project structure..."
mkdir -p /opt/alpha-forge
mkdir -p $PROJECT_PATH/artifacts
mkdir -p $PROJECT_PATH/output

# Step 7: Configure HTTP Basic Auth
step "7" "Configuring HTTP Basic Auth..."
htpasswd -cb /etc/nginx/.htpasswd "$AUTH_USER" "$AUTH_PASS"
chmod 640 /etc/nginx/.htpasswd
chown root:www-data /etc/nginx/.htpasswd

# Save credentials securely
cat > /root/.alpha-forge-credentials << EOF
# Alpha Forge Dashboard Credentials
# Keep this file secure!
URL: http://$(curl -s ifconfig.me)
Username: $AUTH_USER
Password: $AUTH_PASS
EOF
chmod 600 /root/.alpha-forge-credentials

# Step 8: Configure Nginx
step "8" "Configuring Nginx..."
cat > /etc/nginx/sites-available/alpha-forge << 'NGINX'
# Alpha Forge - Professional Backtest Dashboard
server {
    listen 80;
    server_name _;
    
    # Security headers
    add_header X-Frame-Options "SAMEORIGIN" always;
    add_header X-Content-Type-Options "nosniff" always;
    add_header X-XSS-Protection "1; mode=block" always;
    add_header Referrer-Policy "strict-origin-when-cross-origin" always;
    
    # Gzip compression
    gzip on;
    gzip_vary on;
    gzip_min_length 1024;
    gzip_types text/plain text/css application/json application/javascript text/xml application/xml;
    
    # HTTP Basic Auth
    auth_basic "Alpha Forge - Restricted Access";
    auth_basic_user_file /etc/nginx/.htpasswd;
    
    # Dashboard (Vite preview or built files)
    location / {
        proxy_pass http://127.0.0.1:5173;
        proxy_http_version 1.1;
        proxy_set_header Upgrade $http_upgrade;
        proxy_set_header Connection "upgrade";
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
        proxy_read_timeout 86400;
    }
    
    # API endpoints
    location /api/ {
        proxy_pass http://127.0.0.1:3001;
        proxy_http_version 1.1;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
        
        # SSE support
        proxy_set_header Connection '';
        proxy_buffering off;
        proxy_cache off;
        proxy_read_timeout 86400;
    }
    
    # Health check (no auth required)
    location /health {
        auth_basic off;
        proxy_pass http://127.0.0.1:3001/api/health;
    }
}
NGINX

# Enable site
ln -sf /etc/nginx/sites-available/alpha-forge /etc/nginx/sites-enabled/
rm -f /etc/nginx/sites-enabled/default

# Test and reload nginx
nginx -t
systemctl enable nginx
systemctl restart nginx

# Step 9: Configure Firewall
step "9" "Configuring firewall..."
ufw default deny incoming
ufw default allow outgoing
ufw allow ssh
ufw allow http
ufw allow https
ufw --force enable

# Environment variables
cat >> ~/.bashrc << 'EOF'

# Alpha Forge Environment
export DATABASE_URL="postgresql://neondb_owner:npg_HyU68iqJScrQ@ep-wild-cell-af18q8jx-pooler.c-2.us-west-2.aws.neon.tech/neondb?sslmode=require"
export BRAPI_API_KEY="gNJ4vTTpjG8TZJJHGBqoV5"
export RUST_BACKTRACE=1
export CARGO_NET_GIT_FETCH_WITH_CLI=true
export PATH="$HOME/.cargo/bin:$PATH"

# Aliases
alias scg='cargo run -p combiner_cli --release --'
alias bt='cargo run -p backtester_cli --release --'
alias logs='pm2 logs'
alias status='pm2 status'
EOF

source ~/.bashrc

# Final output
echo ""
echo -e "${GREEN}╔═══════════════════════════════════════════════════════════════════╗${NC}"
echo -e "${GREEN}║                    SETUP COMPLETE!                                ║${NC}"
echo -e "${GREEN}╠═══════════════════════════════════════════════════════════════════╣${NC}"
echo -e "${GREEN}║  Dashboard Access:                                                ║${NC}"
echo -e "${GREEN}║  URL:      ${CYAN}http://$(curl -s ifconfig.me)${GREEN}                           ║${NC}"
echo -e "${GREEN}║  Username: ${CYAN}$AUTH_USER${GREEN}                                              ║${NC}"
echo -e "${GREEN}║  Password: ${CYAN}$AUTH_PASS${GREEN}                                   ║${NC}"
echo -e "${GREEN}╠═══════════════════════════════════════════════════════════════════╣${NC}"
echo -e "${GREEN}║  Credentials saved to: /root/.alpha-forge-credentials             ║${NC}"
echo -e "${GREEN}╠═══════════════════════════════════════════════════════════════════╣${NC}"
echo -e "${GREEN}║  Next steps:                                                      ║${NC}"
echo -e "${GREEN}║  1. Build Rust: cd $PROJECT_PATH && cargo build --release        ║${NC}"
echo -e "${GREEN}║  2. Build Dashboard: cd dashboard && npm ci && npm run build     ║${NC}"
echo -e "${GREEN}║  3. Start services: pm2 start ecosystem.config.js                ║${NC}"
echo -e "${GREEN}╠═══════════════════════════════════════════════════════════════════╣${NC}"
echo -e "${GREEN}║  For SSL (requires domain):                                       ║${NC}"
echo -e "${GREEN}║  certbot --nginx -d your-domain.com                              ║${NC}"
echo -e "${GREEN}╚═══════════════════════════════════════════════════════════════════╝${NC}"
echo ""
