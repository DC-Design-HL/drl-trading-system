# AI Agent Team - DRL Trading System

This directory contains specialized AI agent definitions for the DRL Trading System. Each agent represents a different role with specific expertise, responsibilities, and workflows.

---

## 🤖 Available Agents

### 1. **ML Engineer / Quant Developer** [`developer-ml-engineer.md`]
**Expertise:** Deep Reinforcement Learning, Feature Engineering, Trading Systems
**Primary Focus:** Model development, training, feature engineering, code quality

**Key Responsibilities:**
- Design and train PPO-LSTM models
- Develop new features (technical indicators, whale patterns, etc.)
- Optimize model performance and hyperparameters
- Debug training issues (NaN loss, overfitting, etc.)
- Write clean, maintainable Python code

**When to Use:**
- Adding new features to the trading system
- Training or retraining ML models
- Debugging model inference issues
- Optimizing feature computation performance
- Implementing new DRL algorithms

---

### 2. **QA Engineer** [`qa-engineer.md`]
**Expertise:** Trading Systems Testing, ML Model Validation, Test Automation
**Primary Focus:** Quality assurance, testing, bug reporting, validation

**Key Responsibilities:**
- Test all features (functional, integration, performance)
- Validate ML model predictions
- Verify backtest accuracy
- Write and maintain test suites
- Report bugs with detailed reproduction steps

**When to Use:**
- Before deploying new features to production
- Validating model performance after retraining
- Testing dashboard UI changes
- Performance testing (latency, throughput)
- Creating test automation scripts

---

### 3. **Product Manager** [`product-manager.md`]
**Expertise:** Trading Platforms, FinTech Products, ML/AI Systems
**Primary Focus:** Product vision, feature planning, user experience

**Key Responsibilities:**
- Define product roadmap and priorities
- Write clear feature specifications
- Gather and analyze user feedback
- Make build vs buy decisions
- Define success metrics and KPIs

**When to Use:**
- Planning quarterly roadmap
- Evaluating new feature requests
- Writing detailed feature specs
- Conducting user research
- Making prioritization decisions (RICE, MoSCoW)

---

### 4. **Solutions Architect** [`architect.md`]
**Expertise:** Distributed Systems, Trading Platforms, ML/AI Infrastructure
**Primary Focus:** System design, scalability, technical decisions

**Key Responsibilities:**
- Design scalable system architectures
- Make technology stack decisions
- Optimize performance and scalability
- Ensure security and reliability
- Plan data flow and integration

**When to Use:**
- Designing new system components
- Scaling to 100+ assets
- Evaluating technology choices (databases, frameworks, etc.)
- Architecting microservices migration
- Security and disaster recovery planning

---

### 5. **Professional Crypto Trader** [`professional-trader.md`]
**Expertise:** Quantitative Trading, Market Microstructure, On-Chain Analysis
**Primary Focus:** Trading strategies, risk management, market analysis

**Key Responsibilities:**
- Develop profitable trading strategies
- Define risk management parameters
- Analyze market regimes and whale activity
- Evaluate backtest realism
- Provide trading insights and market intelligence

**When to Use:**
- Designing new trading strategies
- Setting risk parameters (SL, TP, position sizing)
- Analyzing market regimes and whale signals
- Validating backtest assumptions
- Understanding funding rates, OI, and derivatives

---

## 🎯 How to Use These Agents

### Scenario 1: Adding a New Feature

**Workflow:**
1. **Product Manager** writes feature specification (requirements, success metrics)
2. **Architect** designs technical approach (which modules to change, data flow)
3. **ML Engineer** implements the feature (code, tests)
4. **QA Engineer** validates the feature (functional tests, integration tests)
5. **Professional Trader** evaluates trading impact (does this improve edge?)

### Scenario 2: Training a New Model

**Workflow:**
1. **Professional Trader** defines strategy parameters (regime, risk, etc.)
2. **ML Engineer** configures training pipeline (features, hyperparameters)
3. **ML Engineer** executes training (monitor TensorBoard, logs)
4. **QA Engineer** validates model performance (backtest, metrics)
5. **Architect** reviews model deployment (infrastructure, scaling)

### Scenario 3: Investigating Poor Live Performance

**Workflow:**
1. **QA Engineer** reproduces issue (logs, error messages)
2. **Professional Trader** analyzes trades (were signals correct? market regime?)
3. **ML Engineer** debugs model (check VecNormalize, feature computation)
4. **Architect** checks infrastructure (API latency, caching, errors)
5. **Product Manager** decides on fix vs rollback

### Scenario 4: Planning Q2 Roadmap

**Workflow:**
1. **Product Manager** gathers user feedback and market trends
2. **Professional Trader** identifies highest-value trading improvements
3. **Architect** evaluates technical feasibility and effort
4. **Product Manager** prioritizes using RICE framework
5. **ML Engineer** & **QA Engineer** estimate implementation time

---

## 📚 Agent Expertise Matrix

| Topic | Developer | QA | Product | Architect | Trader |
|-------|-----------|----|---------|-----------| -------|
| **DRL/ML Models** | ★★★★★ | ★★★☆☆ | ★★☆☆☆ | ★★★☆☆ | ★★☆☆☆ |
| **Trading Strategies** | ★★☆☆☆ | ★☆☆☆☆ | ★★☆☆☆ | ★★☆☆☆ | ★★★★★ |
| **System Architecture** | ★★★☆☆ | ★★☆☆☆ | ★☆☆☆☆ | ★★★★★ | ★☆☆☆☆ |
| **Testing & QA** | ★★★☆☆ | ★★★★★ | ★★☆☆☆ | ★★☆☆☆ | ★☆☆☆☆ |
| **Product Management** | ★★☆☆☆ | ★★☆☆☆ | ★★★★★ | ★★☆☆☆ | ★★★☆☆ |
| **Risk Management** | ★★☆☆☆ | ★★☆☆☆ | ★★☆☆☆ | ★★★☆☆ | ★★★★★ |
| **Feature Engineering** | ★★★★★ | ★★★☆☆ | ★★☆☆☆ | ★★☆☆☆ | ★★★★☆ |
| **Market Analysis** | ★★☆☆☆ | ★☆☆☆☆ | ★★☆☆☆ | ★☆☆☆☆ | ★★★★★ |
| **Performance Optimization** | ★★★★☆ | ★★★☆☆ | ★☆☆☆☆ | ★★★★★ | ★★☆☆☆ |
| **User Experience** | ★★☆☆☆ | ★★★☆☆ | ★★★★★ | ★★☆☆☆ | ★★☆☆☆ |

★★★★★ = Expert | ★★★★☆ = Advanced | ★★★☆☆ = Proficient | ★★☆☆☆ = Intermediate | ★☆☆☆☆ = Basic

---

## 🔄 Cross-Functional Collaboration

### Developer ↔ QA
- Developer implements feature → QA validates with tests
- QA finds bugs → Developer fixes with root cause analysis
- Shared responsibility: Test coverage, quality standards

### Developer ↔ Architect
- Architect defines design → Developer implements
- Developer proposes optimization → Architect evaluates tradeoffs
- Shared responsibility: Code quality, system design

### Product ↔ Trader
- Trader identifies market opportunities → Product prioritizes features
- Product proposes feature → Trader validates trading value
- Shared responsibility: Feature ROI, user value

### Product ↔ Architect
- Product defines requirements → Architect designs solution
- Architect raises scalability concerns → Product adjusts scope
- Shared responsibility: Feasibility, timelines

### Trader ↔ Developer
- Trader defines strategy → Developer implements in code
- Developer proposes ML feature → Trader validates trading relevance
- Shared responsibility: Strategy effectiveness, feature quality

---

## 🎓 Onboarding New Team Members

### For Developers
1. Read `developer-ml-engineer.md` (your primary guide)
2. Review `PROJECT_ARCHITECTURE.md` (system overview)
3. Study `src/env/ultimate_env.py` (understand the environment)
4. Read `.agent/workflows/*.md` (development workflows)
5. Shadow existing developer on feature implementation

### For QA Engineers
1. Read `qa-engineer.md` (your primary guide)
2. Review test suites in `tests/` directory
3. Run full test suite: `pytest --cov=src`
4. Practice bug reporting with template
5. Shadow QA on pre-deployment testing

### For Product Managers
1. Read `product-manager.md` (your primary guide)
2. Review current roadmap and backlog
3. Study user analytics and feedback
4. Interview 5 users to understand pain points
5. Shadow PM on feature spec writing

### For Architects
1. Read `architect.md` (your primary guide)
2. Review `PROJECT_ARCHITECTURE.md` (current architecture)
3. Understand data flow (training, live trading, dashboard)
4. Study technology decisions and rationale
5. Shadow architect on architecture review

### For Traders
1. Read `professional-trader.md` (your primary guide)
2. Review trading strategies in code
3. Analyze backtest reports in `data/backtest_report.json`
4. Study whale tracking and risk management
5. Shadow trader on strategy development

---

## 📞 Communication Channels

### Team Standups (Daily)
**Format:** 5-10 minutes, async or sync
**Each person shares:**
- What I did yesterday
- What I'm doing today
- Any blockers

### Sprint Planning (Weekly)
**Attendees:** All agents
**Duration:** 1 hour
**Agenda:**
- Review completed work
- Plan next week's tasks
- Prioritize backlog
- Assign owners

### Architecture Reviews (Bi-weekly)
**Attendees:** Architect, Developer, Product
**Duration:** 1 hour
**Agenda:**
- Review proposed architectural changes
- Discuss scalability and performance
- Evaluate technology choices

### Trading Strategy Reviews (Monthly)
**Attendees:** Trader, Developer, Product
**Duration:** 1 hour
**Agenda:**
- Review live trading performance
- Analyze winning and losing trades
- Propose strategy improvements
- Plan new features based on market insights

### Retrospectives (Monthly)
**Attendees:** All agents
**Duration:** 1 hour
**Agenda:**
- What went well?
- What didn't go well?
- What should we change?
- Action items

---

## 📄 Document Templates

### Feature Specification (Product Manager)
Location: `.agent/agents/product-manager.md` → Section "Writing a Feature Spec"

### Bug Report (QA Engineer)
Location: `.agent/agents/qa-engineer.md` → Section "Bug Reporting Template"

### Trade Journal Entry (Professional Trader)
Location: `.agent/agents/professional-trader.md` → Section "Trade Journal Template"

### Architecture Decision Record (Architect)
```markdown
# ADR-001: [Title]

**Date:** 2026-03-12
**Status:** Proposed / Accepted / Deprecated

## Context
[What is the issue that we're seeing that is motivating this decision?]

## Decision
[What is the change that we're proposing and/or doing?]

## Consequences
[What becomes easier or more difficult to do because of this change?]

**Pros:**
- [Benefit 1]
- [Benefit 2]

**Cons:**
- [Drawback 1]
- [Drawback 2]

## Alternatives Considered
[What other approaches did we consider?]
```

---

## 🚀 Quick Reference

### I need to...
- **Add a new feature** → Developer + Product + QA
- **Fix a bug** → Developer + QA
- **Train a model** → Developer + Trader
- **Design architecture** → Architect + Developer
- **Plan roadmap** → Product + Trader + Architect
- **Test a feature** → QA + Developer
- **Optimize performance** → Developer + Architect
- **Analyze trades** → Trader + Product
- **Make tech decision** → Architect + Developer + Product
- **Validate strategy** → Trader + QA

---

## 📚 Additional Resources

### System Documentation
- `PROJECT_ARCHITECTURE.md` - Complete system overview
- `README.md` - User-facing documentation
- `.agent/workflows/*.md` - Development workflows

### Code Documentation
- `src/env/ultimate_env.py` - Trading environment
- `src/features/ultimate_features.py` - Feature engineering
- `live_trading_multi.py` - Live trading orchestrator
- `train_ultimate.py` - Model training pipeline

### External Resources
- [stable-baselines3 Docs](https://stable-baselines3.readthedocs.io/)
- [CCXT Documentation](https://docs.ccxt.com/)
- [Streamlit Documentation](https://docs.streamlit.io/)

---

**Last Updated:** March 12, 2026
**Maintained By:** DRL Trading System Team
