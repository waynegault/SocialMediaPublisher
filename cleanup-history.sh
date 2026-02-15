#!/bin/bash
# Git history cleanup using filter-branch
# Run this in Git Bash or WSL

cd /c/Users/wayne/GitHub/Python/Projects/SocialMediaPublisher

# Create backup
cp -r .git .git.backup.$(date +%Y%m%d-%H%M%S)

# Remove sensitive data from history
git filter-branch --force --index-filter \
    'git ls-files -z | xargs -0 sed -i "s/YOUR_GEMINI_API_KEY_HERE/YOUR_GEMINI_API_KEY_HERE/g"' \
    --prune-empty --tag-name-filter cat -- --all

# Clean up
rm -rf .git/refs/original/
git reflog expire --expire=now --all
git gc --prune=now --aggressive

echo "History cleaned. Run: git push --force"
