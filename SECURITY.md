# Security Policy

## Reporting Security Vulnerabilities

If you discover a security vulnerability, please email waynegault@msn.com directly rather than opening a public issue.

## Credential Management

### ⚠️ URGENT: Git History Contains Credentials

**Before using this repository**, be aware that the git history may contain old API keys and credentials. You MUST:

1. **Rotate all API keys** mentioned in the history:
   - Google Gemini API keys
   - LinkedIn OAuth tokens
   - Any other service tokens

2. **Never commit real credentials**:
   ```bash
   # Always use .env.example with placeholders
   cp .env.example .env
   # Edit .env with your REAL credentials (never commit this file)
   ```

3. **Use environment variables** for all sensitive data

### Best Practices

- Keep `.env` in `.gitignore` (it's already there)
- Use different API keys for development and production
- Rotate credentials regularly
- Monitor API usage for unexpected activity

## Supported Versions

| Version | Supported |
|---------|-----------|
| Latest main | ✅ |
| Older commits | ⚠️ Check for credential exposure |

## Security Checklist for Public Repos

Before making any repository public:

- [ ] Rotate all API keys
- [ ] Check git history for credentials (`git log -p | grep -i key`)
- [ ] Ensure .gitignore includes .env, *.key, credentials.json
- [ ] Review SECURITY.md is in place
- [ ] Test with fresh API keys

---

**Last Updated**: February 2026
