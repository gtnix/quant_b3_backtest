# GitHub Upload Safety Checklist

## ✅ Pre-Upload Checklist

Before uploading your project to GitHub, verify these items:

### 1. API Keys and Secrets
- [ ] **NEVER** commit `config/secrets.yaml` (contains your real API keys)
- [ ] ✅ `config/secrets.yaml.example` is included (safe template)
- [ ] ✅ `.gitignore` excludes `config/secrets.yaml`
- [ ] ✅ No API keys hardcoded in any Python files

### 2. Data Files
- [ ] ✅ `data/raw/` directory is excluded (contains downloaded data)
- [ ] ✅ `data/processed/` directory is excluded (contains processed data)
- [ ] ✅ All `.csv`, `.json`, `.parquet` files are excluded
- [ ] ✅ No sample data files accidentally included

### 3. Logs and Reports
- [ ] ✅ All `*.log` files are excluded
- [ ] ✅ `reports/` directory is excluded
- [ ] ✅ No sensitive output files included

### 4. Virtual Environment
- [ ] ✅ `venv/` directory is excluded
- [ ] ✅ `requirements.txt` is included (for dependency management)

### 5. Code Review
- [ ] ✅ No hardcoded API keys in any source files
- [ ] ✅ No database credentials in code
- [ ] ✅ No personal information in comments or strings

## 🔍 Verification Steps

### Check What Will Be Uploaded
```bash
# See what files Git will track
git status

# See what files are ignored
git status --ignored

# Check if secrets file is being tracked (should show as ignored)
git check-ignore config/secrets.yaml
```

### Test the Setup Process
1. Delete your current `config/secrets.yaml`
2. Run the setup script: `python setup_secrets.py`
3. Verify the secrets file is created but not tracked by Git

## 🚀 Upload Process

### 1. Initialize Git Repository
```bash
cd quant_backtest
git init
```

### 2. Add Files (Safe Files Only)
```bash
git add .
```

### 3. Verify What's Being Added
```bash
git status
```

**Expected output should NOT include:**
- `config/secrets.yaml`
- `data/raw/`
- `data/processed/`
- `venv/`
- Any `.log` files

### 4. Create Initial Commit
```bash
git commit -m "Initial commit: Brazilian stock market backtesting engine"
```

### 5. Add Remote Repository
```bash
git remote add origin <your-github-repo-url>
```

### 6. Push to GitHub
```bash
git push -u origin main
```

## 🔒 Security Best Practices

### For Future Development
1. **Always use environment variables or secrets files** for API keys
2. **Never commit secrets files** - they should always be in `.gitignore`
3. **Use the setup script** (`setup_secrets.py`) for new installations
4. **Regularly audit** your repository for accidentally committed secrets

### For Contributors
1. **Fork the repository** instead of cloning directly
2. **Use your own API keys** in your local `secrets.yaml`
3. **Never share your secrets file** in pull requests
4. **Test with the example configuration** before submitting changes

## 🆘 If You Accidentally Commit Secrets

### Immediate Actions
1. **Revoke the API key** immediately on Alpha Vantage
2. **Generate a new API key**
3. **Update your local secrets.yaml** with the new key
4. **Remove the file from Git history**:
   ```bash
   git filter-branch --force --index-filter \
   'git rm --cached --ignore-unmatch config/secrets.yaml' \
   --prune-empty --tag-name-filter cat -- --all
   ```
5. **Force push** to update the remote repository

### Prevention
- Use pre-commit hooks to check for secrets
- Regularly scan your repository for API keys
- Use tools like `git-secrets` or `truffleHog`

## 📋 Final Verification

After uploading, verify these files are **NOT** on GitHub:
- ❌ `config/secrets.yaml`
- ❌ Any data files in `data/raw/` or `data/processed/`
- ❌ Log files
- ❌ Virtual environment files

And verify these files **ARE** on GitHub:
- ✅ All Python source files
- ✅ `config/settings.yaml`
- ✅ `config/secrets.yaml.example`
- ✅ `requirements.txt`
- ✅ `README.md`
- ✅ `.gitignore`

## 🎉 Success!

If all checks pass, your repository is safe to share publicly while keeping your API keys and data private! 