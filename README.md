# Autonomous_AI_Trading_Agent
An autonomous AI trading agent using Deep Reinforcement Learning (PPO). Unlike rule-based bots, it learns profitable strategies via trial-and-error using technical indicators (RSI, SMA). Built with Stable-Baselines3, Gymnasium, and yfinance. Includes full training and backtesting pipelines.
# 📈 Autonomous AI Trading Agent using Deep Reinforcement Learning

![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![Library](https://img.shields.io/badge/Stable--Baselines3-PPO-green)
![Environment](https://img.shields.io/badge/Gymnasium-AnyTrading-orange)
![License](https://img.shields.io/badge/License-MIT-lightgrey)

An end-to-end Machine Learning project that builds an autonomous stock trading bot. Unlike traditional algorithmic traders that use hard-coded rules (e.g., "Buy when RSI < 30"), this agent uses **Deep Reinforcement Learning (PPO)** to learn its own profitable strategies through trial and error.

## 🧠 Project Architecture

* **The Brain (Agent):** Proximal Policy Optimization (PPO) from `stable-baselines3`.
* **The Environment:** A custom wrapper around `gym-anytrading` that simulates a stock exchange.
* **The Vision (Input):** Historical price data enriched with Technical Indicators (RSI, SMA).
* **The Action Space:** Buy, Sell, or Hold.

---

## 🛠️ Installation

1.  **Clone the repository:**
    ```bash
    git clone [https://github.com/yourusername/ai-trading-agent.git](https://github.com/yourusername/ai-trading-agent.git)
    cd ai-trading-agent
    ```

2.  **Install dependencies:**
    ```bash
    pip install gymnasium stable-baselines3 shimmy yfinance pandas-ta matplotlib gym-anytrading
    ```

---

## 🚀 The Workflow (Stage by Stage)

This project follows a 5-stage pipeline to transform raw data into a trained AI agent.

### Stage 1: Data Acquisition
**Goal:** specific historical market data.
We use the `yfinance` API to fetch high-quality daily OHLC (Open, High, Low, Close) data.
* **Default Asset:** `BTC-USD` (Bitcoin).
* **Range:** 2018 to 2024.
* **Process:** The data is cleaned and flattened to ensure compatibility with Pandas.

### Stage 2: Feature Engineering (The Agent's "Vision")
**Goal:** Give the AI context about market trends.
Raw prices alone are often insufficient for an AI to learn patterns. We perform feature engineering using `pandas-ta` to add technical indicators:
* **SMA (Simple Moving Average):** Helps the agent identify the trend direction (Up/Down).
* **RSI (Relative Strength Index):** Helps the agent identify overbought or oversold conditions (Momentum).
* *Why this matters:* This changes the observation space from "What is the price?" to "Is the price high relative to the last 14 days?"

### Stage 3: Environment Setup
**Goal:** Create a game-like simulation for the AI.
We wrap the data into a `Gymnasium` environment. This allows the agent to take actions and receive feedback.
* **Custom Observation:** We override the default environment to ensure the AI sees our custom indicators (RSI/SMA) at every time step.
* **Window Size:** The agent looks at a rolling window of the past **30 days** to make a decision.

### Stage 4: Model Training (PPO)
**Goal:** Teach the agent to maximize profit.
We initialize a **PPO (Proximal Policy Optimization)** agent.
* **Policy:** `MlpPolicy` (Standard Multi-Layer Perceptron Neural Network).
* **Learning:** The agent interacts with the environment for **50,000 timesteps**.
* **Reward Function:** The agent receives a positive reward for increasing portfolio value and a negative reward for losing money.

### Stage 5: Backtesting & Visualization
**Goal:** Verify performance on unseen data.
After training, we switch the environment to use a "Testing" dataset (data the AI hasn't seen before).
* **Execution:** The agent runs through the test data making predictions.
* **Visualization:** A Matplotlib chart is generated showing:
    * **Blue Line:** Stock Price.
    * **Green Dots:** Where the AI bought.
    * **Red Dots:** Where the AI sold.

---

## 💻 Usage

Run the main script to start the entire pipeline:

```bash
python main.py
