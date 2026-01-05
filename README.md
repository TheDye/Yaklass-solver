<div align="center">

# 🧠 AI Homework Solver Bot
### Automated Multi-Platform Assistant

[![Python](https://img.shields.io/badge/Python-3.10%2B-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://python.org)
[![Selenium](https://img.shields.io/badge/Selenium-Automation-43B02A?style=for-the-badge&logo=selenium&logoColor=white)](https://www.selenium.dev/)
[![Perplexity](https://img.shields.io/badge/Powered%20by-Perplexity%20AI-222222?style=for-the-badge&logo=perplexity&logoColor=white)](https://perplexity.ai)
[![Groq](https://img.shields.io/badge/Accelerated%20by-Groq-F55036?style=for-the-badge)](https://groq.com)

<p align="center">
  <img src="https://media.giphy.com/media/v1.Y2lkPTc5MGI3NjExMzM0c3lkM3V4M3V4M3V4M3V4M3V4M3V4M3V4M3V4/26tn33aiTi1jkl6Hz/giphy.gif" width="600" alt="AI Bot Demo">
</p>

*Intelligently solves **Yaklass** and **Google Forms** tests in real-time using a consensus of advanced AI models.*

[Features](#-features) • [Installation](#-installation) • [Usage](#-usage) • [Configuration](#-configuration)

</div>

---

## 📖 About

This bot is a sophisticated automation tool designed to assist with online tests. Instead of relying on a single AI source, it queries **multiple LLMs in parallel** (via Perplexity and Groq), compares their answers using similarity matching, and votes on the best response. It then interacts directly with the browser to type or select the correct answer automatically.

## ✨ Features

- **🧠 Multi-Model Intelligence**: Aggregates answers from **Llama 3**, **Mixtral**, and **Sonar Pro** simultaneously.
- **⚡ Parallel Processing**: Queries all models at once for lightning-fast results (<3s).
- **🗳️ Consensus System**: Uses a voting algorithm to discard hallucinations and ensure high accuracy.
- **🤖 Full Automation**:
  - Auto-detects platform (Yaklass / Google Forms).
  - Handles **Text Inputs** (with human-like typing).
  - Selects **Radio Buttons**, **Checkboxes**, and **Dropdowns**.
  - Auto-submits and navigates to the next question.
- **🎨 Aesthetic CLI**: Beautiful color-coded terminal output with live status updates.
- **🔄 Recursive Solving**: Capable of looping through entire test suites without human intervention.

## 🛠️ Installation

### 1. Prerequisites
- **Python 3.10+**
- **Google Chrome**

### 2. Clone & Install
```bash
# Clone the repository
git clone https://github.com/yourusername/homework-solver-bot.git
cd homework-solver-bot

# Install dependencies
pip install -r requirements.txt
