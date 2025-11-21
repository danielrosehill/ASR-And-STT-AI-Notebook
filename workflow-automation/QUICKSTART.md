# Quick Start Guide - 5 Minutes to Automated Workflow

## 1. Setup (One Time)

```bash
cd /home/daniel/repos/github/STT-Fine-Tuning-Notebook/workflow-automation

# Install dependencies
pip install -r requirements.txt

# Set your API key (add to ~/.bashrc for persistence)
export ANTHROPIC_API_KEY="your-api-key-here"

# Make scripts executable (already done)
chmod +x *.sh
```

## 2. Choose Your Workflow

### Option A: I Want Continuous Auto-Processing (Recommended)

Perfect for active writing sessions where you're adding prompts as you go.

```bash
# Start watch mode (keeps running)
./watch_and_process.sh
```

Now any `.txt` or `.md` file you add to `prompts/to-run/` will be automatically processed!

**Try it:**
```bash
# In another terminal
echo "What are the main challenges in STT fine-tuning?" > ../prompts/to-run/test.txt
```

Watch the first terminal process it automatically!

### Option B: I Have Multiple Prompts Ready

Perfect for batch processing a queue of questions.

```bash
# Process all prompts at once
./batch_process.sh
```

### Option C: I Want to Process One Specific Prompt

```bash
python3 prompt_processor.py --prompt "my-question.txt"
```

## 3. Create Your First Prompt

### Method 1: Interactive (Easy)

```bash
./create_prompt.sh
```

Follow the prompts:
1. Enter topic keywords (e.g., "whisper-quantization")
2. Choose format (.txt or .md)
3. Type/paste your question
4. Press Ctrl+D when done

### Method 2: Manual (Quick)

```bash
echo "What quantization formats are best for edge deployment?" > ../prompts/to-run/$(date +%Y%m%d)-quantization.txt
```

## 4. See the Magic Happen

If watch mode is running, processing happens automatically. Otherwise:

```bash
./batch_process.sh
```

You'll see:
- ✅ Prompt categorized (e.g., "formats")
- ✅ Content generated
- ✅ Saved to category folder
- ✅ Attribution added
- ✅ Original prompt archived

## 5. Check Your Output

```bash
# See what was generated
ls -lh ../formats/  # or whichever category

# Read the generated note
cat ../formats/quantization-edge-deployment.md
```

## Common Workflows

### Daily Note-Taking Session

```bash
# Terminal 1: Start watch mode
./watch_and_process.sh

# Terminal 2: Add prompts as you think of them
echo "Question 1..." > ../prompts/to-run/$(date +%Y%m%d)-topic1.txt
echo "Question 2..." > ../prompts/to-run/$(date +%Y%m%d)-topic2.txt
# etc.

# Watch Terminal 1 process them automatically!
```

### Weekly Batch Processing

```bash
# Accumulate prompts throughout the week in prompts/to-run/
# Then Friday afternoon:

./batch_process.sh

# Commit to git
cd ..
git add .
git commit -m "Weekly STT notebook updates"
git push
```

### One-Off Question

```bash
# Quick question?
./create_prompt.sh
# (enter your question)

# Process immediately
./batch_process.sh
```

## Tips

1. **Name your prompts with dates**: `20251121-topic.txt` - helps with organization
2. **Use watch mode during active sessions** - seamless experience
3. **Use batch mode for bulk processing** - efficient for queues
4. **Review generated content** - always validate technical accuracy
5. **Commit regularly** - keep your notebook version controlled

## File Locations Cheat Sheet

- **Add prompts here**: `prompts/to-run/`
- **Generated content**: `models/`, `formats/`, `training-data/`, etc.
- **Archived prompts**: `prompts/processed/YYYY-MM/`
- **Configuration**: `workflow-automation/config.json`

## Troubleshooting

**"ANTHROPIC_API_KEY not found"**
```bash
export ANTHROPIC_API_KEY="your-key"
```

**Watch mode not working**
```bash
sudo apt install inotify-tools
```

**Want to test without modifying files?**
```bash
python3 prompt_processor.py --dry-run
```

---

That's it! You're now set up for automated STT notebook content generation.

For detailed documentation, see [README.md](./README.md)
