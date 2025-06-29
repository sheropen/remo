# Hierarchical Memory Organization for Wikipedia Generation

## Installation

1. **Create conda environment:**
   ```bash
   conda create -n mog python=3.11.9
   conda activate mog
   ```

2. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   
   # Download NLTK data
   python -c "import nltk; nltk.download('punkt')"
   ```

3. **Set environment variables:**
   ```bash
   cp .env.sample .env
   # Edit .env file to add your API keys:
   # - OPENAI_API_KEY: OpenAI API key
   # - OPENAI_BASE_URL: OpenAI base URL
   # - VALUESERP_API_KEY: ValueSERP API key for web search (register at https://app.valueserp.com/)
   ```

4. **Configuration (optional):**
   - Modify settings in `config/settings.py` for:
     - Model names and token limits
     - Output directories and paths
     - Search and memory parameters
     - Writing depth and clustering thresholds

## Usage

### Generate Articles

**Generate article for a single topic (interactive):**
```bash
python scripts/generate_article.py
```

**Generate article for a specific topic:**
```bash
python scripts/generate_article.py topics.txt
```

**Skip certain phases:**
```bash
python scripts/generate_article.py --skip-research
python scripts/generate_article.py --skip-outline
```

### Refine Articles

**Refine articles from input directory to output directory:**
```bash
python scripts/refine_article.py ./data/input/json ./data/output/refined
```

## Output

Articles are generated in multiple formats:
- **JSON**: Complete structured data
- **Raw Text**: Text with markdown formatting  
- **Clean Text**: Citation-free plain text version with lead section removed for evaluation use