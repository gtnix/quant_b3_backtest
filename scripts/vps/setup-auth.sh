#!/bin/bash
#############################################################################
# Alpha Forge - HTTP Basic Auth Setup
# Creates/updates authentication credentials for dashboard access
#############################################################################

set -e

# Colors
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
CYAN='\033[0;36m'
NC='\033[0m'

# Default values
AUTH_USER="${1:-admin}"
AUTH_PASS="${2:-$(openssl rand -base64 12 | tr -d '/+=' | head -c 16)}"
HTPASSWD_FILE="/etc/nginx/.htpasswd"
CREDS_FILE="/root/.alpha-forge-credentials"

echo ""
echo -e "${CYAN}Alpha Forge - Setting up HTTP Basic Auth${NC}"
echo ""

# Check if running as root
if [ "$EUID" -ne 0 ]; then
    echo "Please run as root"
    exit 1
fi

# Check if apache2-utils is installed
if ! command -v htpasswd &> /dev/null; then
    echo "Installing apache2-utils..."
    apt-get update && apt-get install -y apache2-utils
fi

# Create or update htpasswd file
if [ -f "$HTPASSWD_FILE" ]; then
    echo -e "${YELLOW}Updating existing credentials...${NC}"
    htpasswd -b "$HTPASSWD_FILE" "$AUTH_USER" "$AUTH_PASS"
else
    echo -e "${YELLOW}Creating new credentials file...${NC}"
    htpasswd -cb "$HTPASSWD_FILE" "$AUTH_USER" "$AUTH_PASS"
fi

# Set permissions
chmod 640 "$HTPASSWD_FILE"
chown root:www-data "$HTPASSWD_FILE"

# Get public IP
PUBLIC_IP=$(curl -s ifconfig.me 2>/dev/null || echo "your-server-ip")

# Save credentials to secure file
cat > "$CREDS_FILE" << EOF
# Alpha Forge Dashboard Credentials
# Generated: $(date)
# Keep this file secure!

URL:      http://$PUBLIC_IP
Username: $AUTH_USER
Password: $AUTH_PASS
EOF

chmod 600 "$CREDS_FILE"

# Reload nginx if running
if systemctl is-active --quiet nginx; then
    systemctl reload nginx
    echo -e "${GREEN}Nginx reloaded${NC}"
fi

# Display credentials
echo ""
echo -e "${GREEN}╔═══════════════════════════════════════════════════════════════════╗${NC}"
echo -e "${GREEN}║              Authentication Configured Successfully               ║${NC}"
echo -e "${GREEN}╠═══════════════════════════════════════════════════════════════════╣${NC}"
echo -e "${GREEN}║  URL:      ${CYAN}http://$PUBLIC_IP${NC}"
echo -e "${GREEN}║  Username: ${CYAN}$AUTH_USER${NC}"
echo -e "${GREEN}║  Password: ${CYAN}$AUTH_PASS${NC}"
echo -e "${GREEN}╠═══════════════════════════════════════════════════════════════════╣${NC}"
echo -e "${GREEN}║  Credentials saved to: $CREDS_FILE${NC}"
echo -e "${GREEN}╚═══════════════════════════════════════════════════════════════════╝${NC}"
echo ""

# Usage instructions
echo "Usage:"
echo "  $0                    # Create admin with random password"
echo "  $0 username           # Create user with random password"
echo "  $0 username password  # Create user with specific password"
echo ""











