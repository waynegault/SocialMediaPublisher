# Security Policy

## Reporting Security Issues

If you discover a security vulnerability in this project, please report it by emailing the repository owner. Please do not create public GitHub issues for security vulnerabilities.

## Security Considerations

### API Keys and Credentials

This application requires several API keys and access tokens to function:

- **Google Gemini API Key**: For AI-powered story search and content generation
- **LinkedIn Access Token**: For publishing posts to LinkedIn
- **Hugging Face API Token** (optional): For image generation

**IMPORTANT**: Never commit real API keys or tokens to version control.

### Best Practices

1. **Use Environment Variables**: Store all sensitive credentials in a `.env` file (not tracked by git)
2. **Rotate Keys Regularly**: Periodically regenerate your API keys and tokens
3. **Limit Permissions**: Only grant the minimum required permissions to API tokens
4. **Monitor Usage**: Regularly review API usage logs for unusual activity

### Git History Notice

⚠️ **Important**: If this repository was previously private and is being made public, be aware that git history may contain sensitive information from earlier commits. 

If you had real credentials committed in the past:

1. **Immediately rotate/revoke** any API keys or tokens that may have been exposed
2. Generate new credentials from the respective services:
   - [Google AI Studio](https://aistudio.google.com/) for Gemini API keys
   - [LinkedIn Developer Portal](https://developer.linkedin.com/) for OAuth tokens
   - [Hugging Face](https://huggingface.co/settings/tokens) for API tokens

### Protected Files

The following files are excluded from version control via `.gitignore`:

- `.env` - Contains your actual credentials
- `*.db` - Database files with potentially sensitive data
- `generated_images/` - Generated content
- `data/` and `output/` - Local data directories

### Secure Deployment

For production deployments:

1. Use environment-specific secret management (AWS Secrets Manager, Azure Key Vault, etc.)
2. Enable HTTPS/TLS for all API communications
3. Implement rate limiting to prevent abuse
4. Regularly update dependencies to patch security vulnerabilities
5. Monitor application logs for security events

### Dependencies

Regularly update Python dependencies to ensure security patches are applied:

```bash
pip install --upgrade -r requirements.txt
```

Run security audits on dependencies:

```bash
pip install safety
safety check
```

## Supported Versions

| Version | Supported          |
| ------- | ------------------ |
| 2.x     | :white_check_mark: |
| 1.x     | :x:                |

## Changelog

- **2026-01-07**: Initial public release with sanitized credentials
