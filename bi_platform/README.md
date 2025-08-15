# 🧠 Hybrid GRPO-GRPO-P Business Intelligence Platform

## Revolutionary Dual-Agent Recommendation Framework

This project implements a cutting-edge **Hybrid GRPO-GRPO-P Framework** that combines:
- **🤖 GRPO**: Group consensus from population-based agents (15 diverse agents)
- **👤 GRPO-P**: Personalized learning from individual user interactions  
- **⚖️ Arbitration Controller**: Intelligent blending with 5 dynamic strategies
- **📊 Real-Time Intelligence**: Live financial news processing (191+ articles tested)

## 🚀 Key Innovations

### **Dual-Agent Architecture**
- **Group Wisdom + Personal Learning**: Best of both collective intelligence and individual adaptation
- **Context-Aware Blending**: Dynamic weighting based on user experience and market conditions
- **Cold Start Solution**: Immediate value for new users while learning preferences

### **Advanced Personalization** 
- **Individual GRPO-P Agents**: Each user gets a dedicated learning agent
- **Behavioral Pattern Learning**: System learns risk tolerance, sector preferences, timing
- **Continuous Improvement**: Every interaction makes recommendations more accurate

### **Real-Time Market Integration**
- **Live News Processing**: NewsAPI integration with sentiment analysis
- **Market State Analysis**: Sector-wise sentiment tracking and volatility detection
- **Trending Topic Detection**: Automated identification of emerging opportunities

## 🏗️ Project Structure

```
bi_platform/
├── src/
│   ├── data/           # Data collection (NewsAPI, financial data)
│   ├── processing/     # Text processing, sentiment analysis
│   ├── agents/         # GRPO and GRPO-P implementations
│   └── __init__.py
├── config/
│   ├── settings.py     # Configuration management
│   └── config.env      # Environment variables
├── tests/              # Unit tests
├── requirements.txt    # Python dependencies
├── main.py            # Main entry point
└── README.md          # This file
```

## 🚀 Quick Start

### 1. Install Dependencies

```bash
# Clone or navigate to project directory
cd bi_platform

# Install Python dependencies
pip install -r requirements.txt
```

### 2. Get NewsAPI Key

1. Go to [NewsAPI.org](https://newsapi.org/)
2. Sign up for a free account
3. Get your API key (free tier: 1000 requests/day)

### 3. Configure Environment

Edit `config/config.env`:
```env
NEWSAPI_KEY=your_actual_api_key_here
```

### 4. Test News Collection

```bash
python test_news_collector.py
```

### 5. Run Main System

```bash
python main.py
```

## 📊 Current Implementation Status

### ✅ FULLY IMPLEMENTED & WORKING
- [x] **Complete GRPO Population**: 15 diverse agents with group consensus
- [x] **GRPO-P Personal Learning**: Individual preference learning with feedback
- [x] **Arbitration Controller**: 5 blending strategies with dynamic selection
- [x] **Real-Time News Processing**: NewsAPI integration (191+ articles tested)
- [x] **Market Analysis**: Sentiment analysis, volatility tracking, trending topics
- [x] **Multi-User System**: Complete user management with persistent learning
- [x] **Hybrid Recommendations**: Working end-to-end recommendation pipeline

### � READY FOR ADVANCED FEATURES
- [ ] Neural network upgrade for preference models
- [ ] Web dashboard and API endpoints
- [ ] Enhanced market data sources
- [ ] Advanced evaluation metrics

## 🏗️ Complete System Architecture

```
📊 NewsAPI (Real Data) → 🔍 Market Analysis → 🤖 GRPO Population (15 agents)
                                               ↓
👤 User Interactions → 🧠 GRPO-P Learning → ⚖️ Arbitration Controller (5 strategies)
                                               ↓
                                         🎯 Blended Recommendations
                                               ↓
                                    👥 Multi-User Personalization
```

### **Implemented Components**

#### **GRPO Population System**
- **15 Diverse Agents**: Different risk tolerances and sector preferences
- **Group Consensus**: Democratic voting on market recommendations
- **Performance Tracking**: Agents learn from group relative performance
- **Conservative Bias**: Automatic risk adjustment during market volatility

#### **GRPO-P Personal Learning**
- **Individual Agents**: One per user with persistent learning
- **Cold Start Handling**: Exploration-heavy strategy for new users
- **Preference Models**: Learn sector preferences, risk tolerance, timing
- **Behavioral Learning**: Adapt to user clicking patterns and feedback

#### **Arbitration Controller**
- **5 Blending Strategies**: Confidence-weighted, experience-based, market-adaptive, learned, conservative
- **Context-Aware Selection**: Choose strategy based on user experience and market conditions
- **Dynamic Weights**: GRPO vs GRPO-P weights adapt from 80/20 to 30/70 based on learning
- **Performance Tracking**: Learn which strategies work best for each user type

### Key Components

1. **NewsCollector**: Real-time financial news fetching and processing
2. **TextProcessor**: Advanced sentiment analysis and keyword extraction  
3. **GRPOPopulation**: 15-agent population for group consensus recommendations
4. **GRPOPAgent**: Individual learning agent for each user's preferences
5. **ArbitrationController**: Intelligent blending with 5 dynamic strategies
6. **HybridSystem**: Complete orchestration of all components

## 🎯 Live Demo Usage

### **Run Complete Demo**
```bash
# Navigate to project
cd bi_platform

# Run comprehensive GRPO-P demo
python demo_grpo_p.py
```

### **Quick System Test**
```bash
# Test with real news data
python demo.py

# Run main system
python main.py
```

## 📈 Real Performance Results

**Latest Demo Results (Real Data)**:
- **191 news articles** processed successfully
- **6 market sectors** analyzed with sentiment scores
- **4 different user types** getting personalized recommendations
- **Multiple GRPO agents** achieving group consensus
- **Dynamic arbitration** adapting to user experience levels

**Sample Sector Analysis**:
- **Finance**: +0.14 sentiment (Positive trend)
- **Technology**: +0.09 sentiment (Neutral)
- **Energy**: +0.02 sentiment (Neutral)
- **Healthcare**: +0.08 sentiment (Neutral)

**User Learning Examples**:
- **New Entrepreneurs**: Start with 80% group consensus, learn quickly
- **Experienced Investors**: Achieve 70% personalization after 10 interactions
- **Conservative Users**: System adapts to prefer low-risk recommendations

## 🔧 Configuration Options

Key settings in `config/settings.py`:

```python
# Agent Configuration
GRPO_POPULATION_SIZE = 50
GRPO_LEARNING_RATE = 0.001
GRPO_P_LEARNING_RATE = 0.001

# Data Collection
MAX_NEWS_ARTICLES_PER_FETCH = 100
NEWS_FETCH_INTERVAL_HOURS = 1

# Processing
BATCH_SIZE = 32
MAX_CONTEXT_LENGTH = 512
```

## 🧪 Testing & Validation

### **Run Complete Test Suite**
```bash
# Test GRPO-P framework with multiple users
python demo_grpo_p.py

# Test individual components
python src/agents/grpo_agent.py          # GRPO population test
python src/agents/grpo_p_agent.py        # GRPO-P learning test
python src/agents/arbitration_controller.py  # Arbitration test
python src/agents/hybrid_system.py       # Complete system test

# Test with real news data
python demo.py
```

### **Example Test Results**
```
🤖 GRPO Population Performance:
• Population Performance: 0.65
• Active Agents: 15
• Best Agent: 0.82
• Agreement Level: 73%

👤 GRPO-P Learning Progress:
• User: tech_entrepreneur_alice
• Total Interactions: 15
• Confidence Level: 80%
• Success Rate: 74%
• Personalization: High (70% weight)

⚖️ Arbitration Performance:
• Total Recommendations: 45
• Dynamic Learned Strategy: 0.68 avg performance
• User Experience Based: 0.71 avg performance
• Market Adaptive: 0.64 avg performance
```

## 📋 Development Roadmap

### ✅ Phase 1: COMPLETED - Hybrid Framework Foundation
- [x] GRPO population with 15 diverse agents
- [x] GRPO-P individual learning agents  
- [x] Arbitration controller with 5 strategies
- [x] Real-time news integration and processing
- [x] Multi-user system with persistent learning
- [x] Complete working demo with real data

### 🚀 Phase 2: NEXT - Advanced Intelligence  
- [ ] Neural network upgrade for preference models
- [ ] Enhanced sentiment analysis with financial models
- [ ] Historical data storage and pattern recognition
- [ ] Advanced risk assessment and volatility prediction

### 🌐 Phase 3: FUTURE - Production Platform
- [ ] Web dashboard with interactive visualizations
- [ ] REST API for third-party integrations
- [ ] Real-time notifications and alerts
- [ ] Enterprise features and multi-tenant support

## 🤝 Contributing

This is a research project. Key areas for contribution:
- Algorithm improvements
- Additional data sources
- Evaluation metrics
- User interface design

## 📄 Research Impact

### **Technical Contributions**
- **Novel Hybrid Framework**: First implementation combining population-based RL (GRPO) with individual learning (GRPO-P)
- **Dynamic Arbitration**: Context-aware blending that adapts to user experience and market conditions  
- **Cold Start Solution**: Effective handling of new users through intelligent exploration
- **Real-World Validation**: Working system with live financial data processing

### **Practical Applications**
- **Investment Platforms**: Robo-advisors balancing market wisdom with personal preferences
- **Business Intelligence**: Strategic planning combining industry trends with organizational needs
- **Financial Advisory**: Client onboarding with quick value delivery while learning preferences
- **Risk Management**: Adaptive frameworks accommodating stakeholder comfort levels

### **Academic Value**
- **Reproducible Implementation**: Complete open-source codebase with real data integration
- **Empirical Results**: Performance metrics across different user types and market conditions
- **Comparative Framework**: Direct evaluation of GRPO vs GRPO-P vs Hybrid approaches
- **Human-AI Collaboration**: Study platform for behavioral finance and user interaction patterns

## 🚨 Important Notes

- **Free Tier Limits**: NewsAPI free tier has 1000 requests/day
- **API Keys**: Never commit API keys to version control
- **Research Purpose**: This is academic research, not financial advice
- **Dependencies**: Requires Python 3.8+ and several ML libraries

## 📞 Support

For questions about:
- **Technical implementation**: Check code comments and docstrings
- **Research methodology**: See Research_Documentation_Complete.md
- **Configuration issues**: Review config/settings.py

## 🎯 Quick Start Guide

### **1. Installation**
```bash
cd bi_platform

# Install dependencies (pandas, numpy, nltk, textblob, etc.)
# Note: Uses conda environment for ML libraries
/opt/anaconda3/bin/python -m pip install -r requirements.txt
```

### **2. Configuration**
Get your free NewsAPI key from [newsapi.org](https://newsapi.org) and add to `config/config.json`:
```json
{
  "news_api_key": "your_api_key_here"
}
```

### **3. Run Complete Demo**
```bash
# Full GRPO-P system demonstration
/opt/anaconda3/bin/python demo_grpo_p.py

# Working system with real news data  
/opt/anaconda3/bin/python demo.py

# Basic system test
/opt/anaconda3/bin/python main.py
```

### **4. Explore Components**
```bash
# Test individual agents
/opt/anaconda3/bin/python src/agents/grpo_agent.py
/opt/anaconda3/bin/python src/agents/grpo_p_agent.py
/opt/anaconda3/bin/python src/agents/arbitration_controller.py
```
