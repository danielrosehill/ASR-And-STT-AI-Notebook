# Workflow Automation Architecture

## System Overview

```
┌─────────────────────────────────────────────────────────────────┐
│                    STT Notebook Workflow System                  │
└─────────────────────────────────────────────────────────────────┘

┌──────────────┐      ┌──────────────┐      ┌──────────────┐
│   Input      │      │  Processing  │      │   Output     │
│              │      │              │      │              │
│  prompts/    │─────>│   Claude     │─────>│  Category    │
│  to-run/     │      │   Sonnet     │      │  Folders     │
│              │      │              │      │              │
└──────────────┘      └──────────────┘      └──────────────┘
       │                     │                      │
       │                     │                      │
       v                     v                      v
┌──────────────┐      ┌──────────────┐      ┌──────────────┐
│  Monitoring  │      │ Categorizer  │      │ Attribution  │
│  (inotify)   │      │ (Keywords+AI)│      │ (Badge+Footer)│
└──────────────┘      └──────────────┘      └──────────────┘
       │                                             │
       │                                             │
       v                                             v
┌──────────────┐                            ┌──────────────┐
│   Archive    │                            │  Git Commit  │
│  prompts/    │                            │  (Manual)    │
│  processed/  │                            │              │
└──────────────┘                            └──────────────┘
```

## Component Architecture

### 1. Prompt Processor (`prompt_processor.py`)

**Main orchestrator class**: `PromptProcessor`

```python
PromptProcessor
├── __init__()          # Initialize Anthropic client, load config
├── load_config()       # Load/create category keyword config
├── categorize_prompt() # Determine output category
├── generate_content()  # Create content using Claude
├── generate_filename() # Create appropriate output filename
├── add_attribution()   # Add badge and footer
├── save_content()      # Write to category folder
├── archive_prompt()    # Move to processed folder
├── process_prompt()    # Orchestrate single prompt workflow
└── process_all_prompts() # Batch process all prompts
```

**Dependencies**:
- `anthropic` - Claude API client
- Python 3.9+ standard library (pathlib, datetime, json, re)

### 2. Configuration System (`config.json`)

**Structure**:
```json
{
  "category_keywords": {
    "category-name": ["keyword1", "keyword2", ...]
  },
  "default_category": "ext-ref",
  "naming_conventions": { ... }
}
```

**Purpose**:
- Define keyword → category mappings
- Set default fallback category
- Document naming conventions

### 3. Shell Wrappers

#### `batch_process.sh`
- Simple wrapper around `prompt_processor.py`
- Processes all prompts once
- Returns exit code for scripting

#### `watch_and_process.sh`
- Uses `inotifywait` for file system monitoring
- Detects new files in `to-run` folder
- Triggers processing automatically
- Runs continuously until stopped

#### `create_prompt.sh`
- Interactive prompt creation
- Enforces naming conventions
- Provides user-friendly interface

#### `install_service.sh`
- Installs systemd service
- Configures API key
- Enables auto-start on boot

## Data Flow

### Single Prompt Processing Flow

```
1. Prompt Detection
   ├─> Read file from prompts/to-run/
   └─> Extract prompt text

2. Categorization
   ├─> Check keywords in config
   ├─> If matched → return category
   └─> If not matched → ask Claude

3. Filename Generation
   ├─> Analyze prompt content
   ├─> Ask Claude for concise name
   └─> Clean and format filename

4. Content Generation
   ├─> Send prompt to Claude Sonnet
   ├─> Use specialized system prompt
   └─> Receive markdown content

5. Attribution
   ├─> Prepend badge (if enabled)
   └─> Append footer

6. Save & Archive
   ├─> Write to category-folder/filename.md
   ├─> Handle duplicates (add counter)
   └─> Move original to processed/YYYY-MM/
```

### Category Determination Logic

```python
def categorize(prompt, filename):
    # Step 1: Keyword matching
    for category, keywords in config.items():
        if any(keyword in prompt.lower() for keyword in keywords):
            return category

    # Step 2: Claude analysis (fallback)
    response = claude.ask(categorization_prompt)
    if response.category in valid_categories:
        return response.category

    # Step 3: Default fallback
    return config.default_category
```

## File Naming System

### Input Naming (Prompts)

**Convention**: `YYYYMMDD-topic-keywords.[txt|md]`

**Benefits**:
- Chronological sorting
- Topic identification
- Clear organization

**Example**: `20251121-whisper-quantization.txt`

### Output Naming (Notes)

**Convention**: `topic-keywords.md` (lowercase, hyphenated)

**Benefits**:
- Clean, readable URLs
- Consistent format
- Topic-focused (not date-focused)

**Example**: `whisper-quantization-strategies.md`

**Generation**: Uses Claude to analyze prompt and suggest concise filename

### Archive Naming (Processed Prompts)

**Convention**: `YYYY-MM/original-name_YYYYMMDD-HHMMSS.ext`

**Benefits**:
- Monthly organization
- Preserves original name
- Timestamp prevents conflicts
- Easy to locate by date

**Example**: `2025-11/whisper-quantization_20251121-143022.txt`

## Attribution System

### Badge (Top of File)

```markdown
![Written by Claude](https://img.shields.io/badge/Written%20by-Claude-5A67D8?style=flat-square&logo=anthropic)
```

**Format**: Shields.io badge
**Purpose**: Visual indicator of AI generation
**Placement**: First line of file

### Footer (Bottom of File)

```markdown
---

*Generated by Claude Code - Validate information against current model documentation and benchmarks.*
```

**Format**: Italic disclaimer with horizontal rule
**Purpose**: Remind readers to validate information
**Placement**: After main content

## Monitoring & Automation Options

### Option 1: Manual (Batch)

```bash
./batch_process.sh
```

**Use case**: Periodic processing, controlled workflow

### Option 2: Watch Mode (Interactive)

```bash
./watch_and_process.sh
```

**Use case**: Active writing sessions, immediate processing
**Technology**: `inotifywait` (inotify-tools)
**Events monitored**: `close_write`, `moved_to`

### Option 3: Systemd Service (Background)

```bash
./install_service.sh
sudo systemctl start stt-prompt-processor
```

**Use case**: Always-on background processing
**Technology**: systemd user service
**Features**: Auto-start on boot, restart on failure

## Error Handling

### Categorization Failures

```python
try:
    category = categorize_with_claude(prompt)
except Exception as e:
    log(f"Categorization failed: {e}")
    category = config.default_category  # Fallback
```

**Strategy**: Fall back to default category (`ext-ref`)

### Content Generation Failures

```python
try:
    content = generate_content(prompt)
except Exception as e:
    raise Exception(f"Content generation failed: {e}")
    # Stop processing, preserve prompt
```

**Strategy**: Fail fast, don't archive prompt, allow retry

### Duplicate Filename Handling

```python
if output_path.exists():
    counter = 1
    while output_path.exists():
        output_path = f"{base}-{counter}{ext}"
        counter += 1
```

**Strategy**: Append `-1`, `-2`, etc. to filename

### API Rate Limiting

**Strategy**: Sequential processing (not parallel)
**Benefit**: Avoids rate limit issues
**Trade-off**: Slower batch processing

## Configuration Management

### Default Configuration

Generated automatically if `config.json` doesn't exist:

```python
self.config = {
    "category_keywords": {
        "models": ["model", "whisper", ...],
        # ... other categories
    },
    "default_category": "ext-ref"
}
```

### Custom Configuration

Users can edit `config.json` to:
- Add new keywords
- Change default category
- Adjust naming conventions
- Add new categories (requires code change for folder)

## Extensibility Points

### Adding New Categories

1. Create folder: `mkdir new-category`
2. Add to `CATEGORY_FOLDERS` dict
3. Add keywords to `config.json`

### Changing Attribution Format

Edit constants:
- `ATTRIBUTION_BADGE`
- `ATTRIBUTION_FOOTER`

### Changing AI Model

Edit model parameter in API calls:
```python
model="claude-sonnet-4-5-20250929"
```

### Custom Processing Logic

Override/extend `PromptProcessor` methods:
```python
class CustomProcessor(PromptProcessor):
    def generate_content(self, prompt):
        # Custom logic
        return super().generate_content(prompt)
```

## Performance Characteristics

### Processing Times (Typical)

| Step | Time |
|------|------|
| Categorization (keywords) | <1ms |
| Categorization (Claude) | 1-2s |
| Filename generation | 1-2s |
| Content generation | 5-15s |
| Attribution + Save | <10ms |
| Archive | <10ms |
| **Total** | **10-20s/prompt** |

### Scalability

- **Sequential processing**: One prompt at a time
- **No parallelization**: Avoids API rate limits
- **Memory footprint**: Low (~50MB Python process)
- **Disk I/O**: Minimal (small text files)

### Bottlenecks

1. **Claude API latency**: 5-15s per prompt
2. **Network latency**: 100-500ms per API call
3. **Sequential processing**: No parallelization

**Mitigation**: Use watch mode for real-time processing as prompts are created

## Security Considerations

### API Key Storage

**Environment variable** (recommended):
```bash
export ANTHROPIC_API_KEY="key"
```

**Command line** (less secure):
```bash
python3 prompt_processor.py --api-key "key"
```

**Systemd service** (secure for system service):
```ini
Environment="ANTHROPIC_API_KEY=key"
```

### File Permissions

- Prompt files: User readable/writable
- Output files: User readable/writable
- Scripts: Executable by user
- Service files: Root-owned (standard systemd)

### Prompt Content

**Assumption**: Prompts contain no sensitive data
**Rationale**: All prompts sent to Claude API
**Recommendation**: Don't include secrets/PII in prompts

## Integration Points

### Version Control (Git)

**Manual integration**:
```bash
git add models/ formats/ # etc.
git commit -m "Process STT prompts"
git push
```

**Potential automation**: Git hook on archive folder changes

### CI/CD

Could integrate with:
- GitHub Actions (on push to `prompts/to-run/`)
- GitLab CI (scheduled processing)
- Jenkins (periodic batch jobs)

### Notification Systems

Potential integrations:
- Email notification on completion
- Slack webhook on batch processing
- Desktop notification (notify-send)

## Directory Structure

```
workflow-automation/
├── prompt_processor.py       # Main Python script
├── config.json               # Category keywords config
├── requirements.txt          # Python dependencies
├── batch_process.sh          # Batch processing wrapper
├── watch_and_process.sh      # Continuous monitoring
├── create_prompt.sh          # Interactive prompt creator
├── install_service.sh        # Systemd service installer
├── stt-prompt-processor.service  # Systemd unit file
├── README.md                 # Comprehensive documentation
├── QUICKSTART.md            # 5-minute getting started
└── ARCHITECTURE.md          # This file
```

## Future Enhancement Ideas

### Short-term
- [ ] Desktop notifications on completion
- [ ] Configurable attribution templates
- [ ] Support for prompt templates
- [ ] Batch categorization preview
- [ ] Progress bar for batch processing

### Medium-term
- [ ] Web interface for prompt submission
- [ ] Git auto-commit on completion
- [ ] Email digest of processed prompts
- [ ] Multiple AI model support
- [ ] Prompt queue prioritization

### Long-term
- [ ] Multi-user support
- [ ] API server mode
- [ ] Integration with note-taking apps
- [ ] Automatic cross-referencing between notes
- [ ] Vector database for semantic search

---

**Design Principles**:
1. **Simplicity**: Easy to understand and modify
2. **Modularity**: Components can be used independently
3. **Flexibility**: Multiple usage modes (batch, watch, service)
4. **Reliability**: Error handling and fallback strategies
5. **Transparency**: Clear logging and feedback
