# Git History Cleanup - PowerShell Version
# Uses git filter-repo (if available) or git filter-branch

param(
    [switch]$ForcePush
)

$repoPath = "C:\Users\wayne\GitHub\Python\Projects\SocialMediaPublisher"
Push-Location $repoPath

try {
    Write-Host "🧹 Git History Cleanup" -ForegroundColor Cyan
    Write-Host "======================" -ForegroundColor Cyan
    
    # Check for git-filter-repo (modern tool)
    $filterRepo = Get-Command git-filter-repo -ErrorAction SilentlyContinue
    
    if ($filterRepo) {
        Write-Host "Using git-filter-repo (modern tool)" -ForegroundColor Green
        
        # Create replacement expressions file
        $replacements = @"
YOUR_GEMINI_API_KEY_HERE==>YOUR_GEMINI_API_KEY_HERE
YOUR_LINKEDIN_TOKEN_HERE==>YOUR_LINKEDIN_TOKEN_HERE
"@
        $replacements | Out-File -FilePath "$env:TEMP\git-replacements.txt" -Encoding utf8
        
        git filter-repo --replace-text "$env:TEMP\git-replacements.txt" --force
    } else {
        Write-Host "git-filter-repo not found, using git filter-branch (legacy)" -ForegroundColor Yellow
        
        # Backup
        Copy-Item .git "..\SocialMediaPublisher.git.backup.$(Get-Date -Format 'yyyyMMdd-HHmmss')" -Recurse
        
        # Use filter-branch with sed-like replacement
        # This is complex on Windows, so we'll use a different approach
        
        Write-Host "Installing git-filter-repo via pip..." -ForegroundColor Yellow
        pip install git-filter-repo
        
        # Now use it
        $replacements = @"
YOUR_GEMINI_API_KEY_HERE==>YOUR_GEMINI_API_KEY_HERE
YOUR_LINKEDIN_TOKEN_HERE==>YOUR_LINKEDIN_TOKEN_HERE
"@
        $replacements | Out-File -FilePath "$env:TEMP\git-replacements.txt" -Encoding utf8
        
        git filter-repo --replace-text "$env:TEMP\git-replacements.txt" --force
    }
    
    # Clean up
    Write-Host "Cleaning up..." -ForegroundColor Cyan
    git reflog expire --expire=now --all
    git gc --prune=now --aggressive
    
    Write-Host "✅ History cleaned!" -ForegroundColor Green
    
    if ($ForcePush) {
        Write-Host "🚀 Force pushing..." -ForegroundColor Cyan
        git push --force
        Write-Host "✅ Force push complete!" -ForegroundColor Green
    } else {
        Write-Host "" 
        Write-Host "⚠️  Review changes, then run:" -ForegroundColor Yellow
        Write-Host "   git push --force" -ForegroundColor Cyan
    }
    
} finally {
    Pop-Location
}
