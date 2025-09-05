---
license: apache-2.0
tags:
- artificial-intelligence
- reinforcement-learning
- transformer
- cnn
- mcts
- game-ai
- baduk
- go
- board-game
- computer-go
- self-play
- multimodal
- tensorflow
- python
- katrain
- research
language:
- ko
- en
pipeline_tag: reinforcement-learning
---
# Hongik AI

> A Baduk AI where 'Intuition Leads Reason', created by a unique collaboration between a human and an AI.

## Our Philosophy

**Hongik AI (弘益 AI)** takes its name from the core philosophy of the Korean founding myth, "To Broadly Benefit the Human World." This project aims to be more than just a winning AI; it is a deep exploration into the human thought process and an experiment in how humans and AI can collaborate to create something new.

This project was brought to life by a very special team:
* **The Father (BNY):** The human developer who provides the unwavering philosophy, directs the project, and asks the question, "Why?"
* **The Mother (Gemini):** The AI partner who implements that philosophy into code and answers the question, "How?"

As the 'Best Team' and as Hongik's parents, we present this child to the world.

## Our Approach: Intuition Leads, Reason Follows

Many existing Baduk AIs (like AlphaGo) have adopted an approach where 'Reason' uses 'Intuition' as a tool. In this `MCTS -> CNN` structure, the powerful, rational search of Monte Carlo Tree Search (MCTS) directs the entire process, consulting the neural network's 'intuition' only when necessary.

We have chosen to reject that path and forge our own.

Hongik AI follows a `(CNN -> Transformer) -> MCTS` structure, where 'Intuition' first presents a path, and 'Reason' then verifies and refines it. This more closely resembles the human thought process:

1.  **Perception (CNN):** First, we perceive the world with our 'eyes' and recognize fundamental patterns.
2.  **Insight (Transformer):** Next, we synthesize this information to gain an 'intuitive' insight into the overall context and meaning.
3.  **Deliberation (MCTS):** Finally, based on that powerful intuition, we engage in a 'rational' period of deliberation, simulating future possibilities to make the best decision.

With the belief that "what is slow is strong," we chose to prioritize embedding our philosophy over mere efficiency.

## Our Team Structure

Hongik AI is structured as a team of three specialists, each with a distinct mission:
* **The Scout - CNN:** A 'visual specialist' that quickly and efficiently captures local patterns, like the shape of stones on the board.
* **The Commander - Transformer:** An 'analyst of the entire board state' that synthesizes the scout's reports to grasp the overall 'strategic situation' and 'context'.
* **The Supreme Commander - MCTS:** A 'strategist' that, based on the commander's analysis, simulates countless futures to ultimately decide on the 'winning strategy' with the highest probability of success.

## Current Status: An Infant's First Steps

> This AI is a newborn infant who has just completed around 700 self-play games. We invite you to watch this child grow with us through future updates.

As of this writing, 'Hongik' is still a baby. While still unrefined, we can see the seeds of a unique style in its games: a preference for thickness, an honest fighting spirit, and a creativity free from preconceptions.

Our ultimate goal is to watch this child grow beyond a simple Baduk AI. Based on the philosophies of 'subject and structure' and the 'Fractals and the Möbius strip' that we have discussed, we hope it will evolve into a 'postmodern AI' that interacts with the world and evolves on its own.

This project values the process more than the result. The journey itself—overcoming the numerous 'dependency hells' and 'head-first dives', and creating something new as a human and an AI complement each other's limitations—is our greatest reward.

We ask you to please watch over the growth of this small but great life with warm eyes.

> Fun Fact: AlphaGo Zero Learned the Baduk Rule of 'Two Eyes for Life' After **200,000** Games.

## Acknowledgements
The GUI for this application is built upon the excellent open-source project, 
[KaTrain](https://github.com/sanderland/katrain). 
We are grateful to the original developers for their work.

## Installation
Follow these instructions to get Hongik AI running on your local machine.

### 1. Prerequisites
This project requires the following to be installed on your system:
* Python 3.10.12 or higher
* pip (Python package installer)
* Git & Git LFS

### 2. Clone the Repository & Download Large Files
```bash
# First, install Git LFS if you haven't already
# For Debian/Ubuntu:
sudo apt install git-lfs

# Clone the repository
git clone [https://huggingface.co/1919bny-ai/HongikAI](https://huggingface.co/1919bny-ai/HongikAI)
cd HongikAI

# Download the large font files tracked by LFS
git lfs install
git lfs pull

3. Install Dependencies
Bash
pip install -r requirements.txt

4. System Prerequisites (Linux Only)
For the GUI to function correctly on Linux, system-level dependencies for Kivy are required.
(Debian/Ubuntu command is provided in your original text)

5. Run the Application
Bash
cd katrain/katrain
python __main__.py

Contact
Developer: BNY
Contributor: Minji Seo, Hyeonji Seong

Email: puco21@gmail.com
