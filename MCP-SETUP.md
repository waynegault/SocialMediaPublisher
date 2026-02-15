# Shared MCP Server Configuration

This document describes the shared MCP (Model Context Protocol) server setup between GitHub Copilot (VS Code) and OpenClaw agents.

## Overview

Both systems now share the same MCP servers:
- **GitHub Copilot** (in VS Code) → uses `.vscode/mcp.json`
- **OpenClaw Agents** (Hal, Codex, etc.) → uses `mcporter` with shared config

This creates a unified tool ecosystem where both AI assistants can access the same capabilities.

## MCP Servers Configured

### 1. GitHub MCP Server
**Purpose**: Access GitHub repositories, issues, PRs

**Tools available**:
- `create_or_update_file` - Create/update files in repos
- `search_repositories` - Search GitHub repos
- `create_repository` - Create new repos
- `get_file_contents` - Read file contents
- `create_issue` - Create issues
- `create_pull_request` - Create PRs
- `fork_repository` - Fork repos
- `search_code` - Search code across GitHub
- `list_commits` - List commits
- `get_issue` - Get issue details
- `update_issue` - Update issues
- `add_issue_comment` - Comment on issues
- `list_pull_requests` - List PRs
- `get_pull_request` - Get PR details
- `create_pull_request_review` - Review PRs
- `merge_pull_request` - Merge PRs
- `get_pull_request_files` - Get PR changed files
- `get_pull_request_status` - Check PR status
- `update_pull_request_branch` - Update PR branch
- `get_pull_request_comments` - Get PR comments
- `get_pull_request_reviews` - Get PR reviews

**Usage in Copilot**:
```
Search for repos about MCP servers
Create an issue about bug fix
```

**Usage in OpenClaw**:
```
mcporter call github.search_repositories query="MCP servers"
mcporter call github.create_issue owner="waynegault" repo="MyRepo" title="Bug fix"
```

### 2. Filesystem MCP Server
**Purpose**: Read/write files on local system

**Tools available**:
- `read_file` - Read file contents
- `read_multiple_files` - Read multiple files
- `write_file` - Write/create files
- `create_directory` - Create directories
- `list_directory` - List directory contents
- `move_file` - Move/rename files
- `search_files` - Search file contents
- `get_file_info` - Get file metadata

**Root directory**: `C:\Users\wayne\GitHub`

**Usage in Copilot**:
```
Read the README.md file
List all Python files in this project
```

**Usage in OpenClaw**:
```
mcporter call filesystem.read_file path="README.md"
mcporter call filesystem.list_directory path="."
```

### 3. Web Search MCP Server (Brave)
**Purpose**: Search the web

**Tools available**:
- `brave_web_search` - Search the web
- `brave_local_search` - Local business search

**Usage in Copilot**:
```
Search for MCP server documentation
Find Python best practices for 2026
```

**Usage in OpenClaw**:
```
mcporter call web-search.brave_web_search query="MCP servers"
```

### 4. Ancestry MCP Server (OpenClaw-only for now)
**Purpose**: Access ancestry/genealogy database

**Tools available**:
- `query_database` - Query ancestry database
- `get_match_summary` - Get DNA match summary
- `search_matches` - Search for matches
- etc.

**Usage in OpenClaw**:
```
mcporter call ancestry.query_database query="SELECT * FROM matches"
```

## Setup Instructions

### For VS Code / GitHub Copilot

1. **Ensure VS Code 1.99+**:
   ```bash
   code --version
   ```

2. **Enable MCP in Copilot settings**:
   - Open VS Code Settings
   - Search "GitHub Copilot"
   - Enable "MCP servers"

3. **The `.vscode/mcp.json` is already configured** in your repo

4. **Restart VS Code**

5. **Test in Copilot Chat**:
   ```
   @workspace List the files in this directory
   ```

### For OpenClaw Agents

1. **mcporter is already configured** with the same servers

2. **Test connection**:
   ```bash
   mcporter list
   ```

3. **Call MCP tools from agents**:
   ```
   @coding Use the GitHub MCP to create an issue about this bug
   @archivist Search the web for latest MCP developments
   ```

## Environment Variables

These need to be set for MCP servers to work:

```bash
# GitHub MCP
export GITHUB_TOKEN="your_github_personal_access_token"

# Brave Search MCP  
export BRAVE_API_KEY="your_brave_api_key"
```

Add to your `.env` file or system environment.

## Security Considerations

- MCP servers have access to your filesystem/GitHub
- Tokens are stored in environment variables (not committed)
- Each server runs in its own process
- Copilot and OpenClaw use the same server instances

## Troubleshooting

**MCP server not starting**:
```bash
# Check if npx is available
npx --version

# Install Node.js if needed
# https://nodejs.org/
```

**Permission errors**:
- Ensure `GITHUB_TOKEN` has necessary scopes
- Filesystem MCP is limited to `C:\Users\wayne\GitHub`

**Server not showing in Copilot**:
- Check VS Code version (must be 1.99+)
- Restart VS Code after config changes
- Check MCP is enabled in settings

## Benefits

1. **Unified Tools**: Same capabilities in both Copilot and OpenClaw
2. **Consistency**: Both AIs see the same data
3. **Flexibility**: Choose the right AI for the task
4. **Extensibility**: Add more MCP servers as needed

## Future Enhancements

Potential additional MCP servers:
- **PostgreSQL** - Database access
- **Slack** - Team communication
- **Notion** - Documentation/knowledge base
- **Linear** - Issue tracking
- **Browser** - Web automation
- **Memory** - Shared context between sessions
