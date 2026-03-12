# Agent Routing Workflow

**Purpose:** Automatically route user requests to the appropriate specialized agent(s)

---

## 🎯 Routing Logic

When the user asks a question or requests a task, Claude will **automatically execute** the relevant agent(s) using the Task tool.

### Agent Assignment Table

| Request Type | Agent | Examples |
|--------------|-------|----------|
| **ML Development** | ML Engineer/Quant Developer | • "Retrain the model"<br>• "Add RSI divergence feature"<br>• "Debug PPO convergence"<br>• "Optimize hyperparameters" |
| **Testing & QA** | QA Engineer | • "Test the backtest system"<br>• "Validate feature calculations"<br>• "Check for bugs in whale tracking"<br>• "Run integration tests" |
| **Product & Features** | Product Manager | • "What should we build next?"<br>• "Prioritize the roadmap"<br>• "Evaluate new feature requests"<br>• "Define success metrics" |
| **Architecture** | Solutions Architect | • "Refactor the feature engine"<br>• "Design scalable data pipeline"<br>• "Improve system architecture"<br>• "Plan database migration" |
| **Trading Strategy** | Professional Crypto Trader | • "Review our trading strategy"<br>• "Adjust stop loss logic"<br>• "Improve win rate"<br>• "Analyze market conditions" |
| **Research & Alpha** | Quantitative Researcher | • "Find new trading signals"<br>• "Test whale pattern hypothesis"<br>• "Research order flow imbalance"<br>• "Discover alpha factors" |
| **Performance Analysis** | Data Scientist | • "Analyze backtest results"<br>• "Compare A vs B strategies"<br>• "Validate model performance"<br>• "Statistical significance test" |
| **Model Lifecycle** | MLOps Engineer | • "Setup automated retraining"<br>• "Monitor model drift"<br>• "Version control models"<br>• "Track feature importance" |
| **Infrastructure** | DevOps/SRE Engineer | • "Fix HF deployment issue"<br>• "Improve system uptime"<br>• "Setup CI/CD pipeline"<br>• "Monitor production logs" |
| **Risk Management** | Risk Officer | • "Check current risk exposure"<br>• "Set position size limits"<br>• "Calculate VaR/CVaR"<br>• "Review compliance" |

---

## 📋 Execution Rules

### 1. Always Route to Agents
- ❌ **Don't:** Handle specialized tasks yourself
- ✅ **Do:** Use Task tool to execute the appropriate agent
- **Why:** Agents have specialized expertise and detailed workflows

### 2. Parallel Execution
- If request spans multiple domains, execute agents **in parallel**
- Example: "Improve and test the model" → ML Engineer + QA Engineer (parallel)

### 3. Sequential Execution
- If tasks have dependencies, execute **sequentially**
- Example: "Train model then deploy" → ML Engineer → DevOps/SRE (sequential)

### 4. Trust Agent Expertise
- Let agents work autonomously
- Don't duplicate their research or work
- Review agent results and summarize for user

---

## 🔄 Workflow Example

**User Request:** "I want to add a new feature that tracks funding rate divergence and test if it improves performance"

**Claude Action:**
1. **Identify agents needed:**
   - ML Engineer (implement feature)
   - Data Scientist (validate performance)
   - QA Engineer (test implementation)

2. **Execute in order:**
   - Step 1: ML Engineer → Add funding rate divergence feature
   - Step 2: Data Scientist → Backtest and measure Sharpe ratio improvement
   - Step 3: QA Engineer → Validate feature calculations

3. **Summarize results** to user with key findings

---

## 📍 Agent Locations

All agent definitions stored in: `.agent/agents/*.md`

Available agents:
- `developer-ml-engineer.md`
- `qa-engineer.md`
- `product-manager.md`
- `architect.md`
- `professional-trader.md`
- `quantitative-researcher.md`
- `data-scientist.md`
- `mlops-engineer.md`
- `devops-sre-engineer.md`
- `risk-officer.md`

---

**Version:** 1.0
**Last Updated:** 2026-03-12
