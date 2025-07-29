# ChatGPT Prompt for Telugu Stock Analysis

## System Message:
You are an expert Indian stock market analyst with deep knowledge of Telugu financial channels. Provide detailed, actionable analysis from video transcripts. Always respond with valid JSON format.

## User Prompt Template:

You are an expert Indian stock market analyst. Analyze this Telugu YouTube video transcript for investment insights.

Channel: {channel}
Title: {video_title}

Video Metadata:
- Duration: {duration} seconds
- Views: {view_count}
- Upload Date: {upload_date}

Transcript:
{transcript_text}

Analyze the content and extract:

1. **Specific stock recommendations** with company names (NSE/BSE symbols if mentioned)
2. **Price targets, percentage gains, or numerical predictions**
3. **Market sentiment and reasoning**
4. **Investment timeframe** (short/medium/long term)
5. **Risk factors or warnings** mentioned
6. **Sector analysis or themes**
7. **Technical analysis signals** (support, resistance, breakouts, etc.)

Focus on extracting actionable investment information. If no clear recommendations exist, indicate that in the response.

Respond in this exact JSON format:
```json
{
  "stock_recommendations": [
    {
      "symbol": "STOCK_SYMBOL",
      "company_name": "Full Company Name",
      "action": "BUY/SELL/HOLD",
      "target_price": "if mentioned or null",
      "rationale": "reason for recommendation",
      "confidence": 0.0-1.0,
      "timeframe": "short/medium/long"
    }
  ],
  "market_sentiment": "BULLISH/BEARISH/NEUTRAL",
  "confidence_score": 0.0-1.0,
  "key_themes": ["theme1", "theme2", "theme3"],
  "sector_insights": {
    "IT": {"mentions": 0, "sentiment": "NEUTRAL", "keywords_found": []},
    "Banking": {"mentions": 0, "sentiment": "NEUTRAL", "keywords_found": []}
  },
  "technical_signals": {
    "support": {"found": false, "keywords": [], "context": []},
    "resistance": {"found": false, "keywords": [], "context": []}
  },
  "risks_mentioned": ["risk1", "risk2"],
  "price_targets_mentioned": ["target1", "target2"],
  "numerical_predictions": ["prediction1", "prediction2"]
}
```

Make sure the JSON is valid and complete. If a section has no relevant information, use empty arrays or appropriate null values.

---

## Example Usage:

### Sample Input:
Channel: moneypurse
Title: Next 10 years లో ఈ 10 stocks భారీగా పెరగనున్నాయా? TatvaChintan 500% profit growth reversal started?
Duration: 1667 seconds
Views: 44136
Upload Date: 20250725

Transcript: "How To Make An Post... [transcript content]..."

### Expected Output:
```json
{
  "stock_recommendations": [
    {
      "symbol": "TATVACHINTAN",
      "company_name": "Tatva Chintan Pharma Chem",
      "action": "BUY",
      "target_price": null,
      "rationale": "500% profit growth reversal mentioned for next 10 years",
      "confidence": 0.7,
      "timeframe": "long"
    }
  ],
  "market_sentiment": "BULLISH",
  "confidence_score": 0.8,
  "key_themes": ["long term growth", "pharma sector", "profit growth reversal"],
  "sector_insights": {
    "Pharma": {"mentions": 3, "sentiment": "BULLISH", "keywords_found": ["TatvaChintan", "pharma", "growth"]},
    "IT": {"mentions": 1, "sentiment": "NEUTRAL", "keywords_found": ["software"]}
  },
  "technical_signals": {
    "support": {"found": true, "keywords": ["base"], "context": ["base formation visible"]},
    "resistance": {"found": false, "keywords": [], "context": []}
  },
  "risks_mentioned": [],
  "price_targets_mentioned": ["500% growth"],
  "numerical_predictions": ["500% profit growth in next 10 years"]
}
```

---

## Integration with Daily Stock News Agent

This prompt is now automatically used by the **Content Analysis Tool** when OpenAI API is configured:

1. **Environment Setup**: Set `OPENAI_API_KEY` environment variable
2. **Tool Configuration**: Content Analysis Tool automatically uses this prompt format
3. **Fallback**: If OpenAI fails, falls back to pattern-based analysis
4. **Output**: Generates structured analysis in `./data/analyses/{date}/{channel}_analysis.json`

**Note**: The automated system truncates transcripts to 4000 characters for token efficiency while maintaining context.

## Setting Up OpenAI API

To enable LLM analysis in the Daily Stock News Agent:

### 1. Get OpenAI API Key
- Visit [OpenAI API Keys](https://platform.openai.com/api-keys)
- Create a new API key
- Copy the key (starts with `sk-`)

### 2. Set Environment Variable
```bash
# For macOS/Linux (add to ~/.bashrc or ~/.zshrc for persistence)
export OPENAI_API_KEY="sk-your-api-key-here"

# For Windows PowerShell
$env:OPENAI_API_KEY="sk-your-api-key-here"

# Or create a .env file in the project root
echo "OPENAI_API_KEY=sk-your-api-key-here" > .env
```

### 3. Verify Setup
```bash
cd /Users/saipraveen/Gen-AI/Daily-Stock-News-Agent
python -c "import os; print('API Key configured:', bool(os.getenv('OPENAI_API_KEY')))"
```

### 4. Test LLM Analysis
```bash
# Run the complete workflow - it will automatically use OpenAI for analysis
python framework_comparison/implementations/swarm_agent.py
```

## Benefits of LLM Analysis vs Pattern-Based

| Feature | Pattern-Based | LLM-Based |
|---------|---------------|-----------|
| **Stock Recognition** | Limited keywords | Deep understanding |
| **Context Understanding** | Basic patterns | Semantic analysis |
| **Recommendation Extraction** | Simple rules | Complex reasoning |
| **Confidence Scoring** | Fixed rules | Dynamic assessment |
| **Sector Analysis** | Keyword matching | Contextual insights |
| **Technical Analysis** | Pattern detection | Comprehensive signals |
| **Multi-language Support** | Basic | Advanced Telugu understanding |

## Manual Testing with ChatGPT

For manual testing or analysis outside the automated system:

### 1. Copy the System Message and User Prompt above
### 2. Replace `{transcript}` with actual video transcript
### 3. Submit to ChatGPT and get structured JSON analysis

## Troubleshooting

### Common Issues:
1. **"LLM analysis failed"** - Check API key and internet connection
2. **"Using local pattern-based analysis"** - API key not configured
3. **Rate limits** - OpenAI API has usage limits, tool automatically falls back
4. **Invalid JSON** - Tool handles malformed responses gracefully

### Debug Mode:
```bash
# Enable debug logging to see LLM interactions
export DEBUG_LLM=true
python framework_comparison/implementations/swarm_agent.py
```

## Cost Optimization

The Content Analysis Tool optimizes OpenAI API costs by:
- **Truncating transcripts** to 4000 characters (keeps essential content)
- **Using GPT-4** for accuracy (more expensive but better results)
- **Low temperature** (0.1) for consistent responses
- **Fallback system** to avoid unnecessary API calls on failures
- **Caching results** in JSON files to avoid re-analysis

Estimated cost per analysis: **$0.10 - $0.30** depending on transcript length.

## Manual Testing with ChatGPT

For manual testing or analysis outside the automated system:

### 1. Copy the System Message and User Prompt above
### 2. Replace `{transcript}` with actual video transcript  
### 3. Submit to ChatGPT and get structured JSON analysis

### Example Manual Test:

**System Message:** (Use the system message above)

**User Prompt:**
```
You are an expert Indian stock market analyst. Analyze this Telugu YouTube video transcript for investment insights.

Channel: moneypurse
Title: Next 10 years లో ఈ 10 stocks భారీగా పెరగనున్నాయా? TatvaChintan 500% profit growth reversal started?

Video Metadata:
- Duration: 1667 seconds
- Views: 44136
- Upload Date: 20250725

Transcript:
[Insert actual cleaned transcript here - the automated system handles this automatically]

Analyze the content and extract:

1. **Specific stock recommendations** with company names (NSE/BSE symbols if mentioned)
2. **Price targets, percentage gains, or numerical predictions**
3. **Market sentiment and reasoning**
4. **Investment timeframe** (short/medium/long term)
5. **Risk factors or warnings** mentioned
6. **Sector analysis or themes**
7. **Technical analysis signals** (support, resistance, breakouts, etc.)

Focus on extracting actionable investment information. If no clear recommendations exist, indicate that in the response.

Respond in this exact JSON format:
{
  "stock_recommendations": [
    {
      "symbol": "STOCK_SYMBOL",
      "company_name": "Full Company Name",
      "action": "BUY/SELL/HOLD",
      "target_price": "if mentioned or null",
      "rationale": "reason for recommendation",
      "confidence": 0.0-1.0,
      "timeframe": "short/medium/long"
    }
  ],
  "market_sentiment": "BULLISH/BEARISH/NEUTRAL",
  "confidence_score": 0.0-1.0,
  "key_themes": ["theme1", "theme2", "theme3"],
  "sector_insights": {
    "IT": {"mentions": 0, "sentiment": "NEUTRAL", "keywords_found": []},
    "Banking": {"mentions": 0, "sentiment": "NEUTRAL", "keywords_found": []}
  },
  "technical_signals": {
    "support": {"found": false, "keywords": [], "context": []},
    "resistance": {"found": false, "keywords": [], "context": []}
  },
  "risks_mentioned": ["risk1", "risk2"],
  "price_targets_mentioned": ["target1", "target2"],
  "numerical_predictions": ["prediction1", "prediction2"]
}

Make sure the JSON is valid and complete. If a section has no relevant information, use empty arrays or appropriate null values.
```

### Tips for Manual Analysis:

1. **Use current transcripts** - Get fresh transcripts from recent videos for best results
2. **Include metadata** - Channel name, video title, and date provide important context
3. **Check JSON validity** - Ensure the response is valid JSON before processing
4. **Verify stock symbols** - Cross-check any stock symbols mentioned against NSE/BSE listings
5. **Note confidence scores** - Higher confidence indicates more reliable recommendations

## Automated vs Manual Workflow

| Aspect | Automated System | Manual ChatGPT |
|--------|------------------|-----------------|
| **Trigger** | Run SwarmAgent workflow | Copy-paste prompts manually |
| **Input** | Video URLs, channel configs | Manual transcript input |
| **Processing** | Handles transcription, analysis, reporting | Manual prompt engineering |
| **Output** | JSON files in date-organized folders | Raw ChatGPT response |
| **Consistency** | Standardized format, automated validation | Manual formatting needed |
| **Scale** | Process multiple videos automatically | One-by-one manual analysis |
| **Integration** | Feeds into report generation system | Standalone analysis |

## System Architecture

The Daily Stock News Agent now operates entirely on **LLM-based analysis**:

- **No Rule-Based Fallback**: System requires OpenAI API configuration
- **Advanced Prompt Engineering**: Comprehensive prompts for structured analysis
- **Quality Assurance**: LLM provides better context understanding than pattern matching
- **Scalable**: Handles complex Telugu financial content with semantic understanding
- **Reliable**: Consistent JSON output format for automated processing

This approach ensures high-quality analysis that can understand context, nuance, and domain-specific terminology that rule-based systems cannot match.
